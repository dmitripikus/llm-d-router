/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package sloattainment provides a response plugin that classifies each completed request
// against its TTFT and TPOT targets and records SLO attainment per priority, prompt-length
// bucket and prefill/decode path.
package sloattainment

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// PluginType is the registered plugin type.
	PluginType = "slo-attainment-reporter"

	defaultPrefillProfile = "prefill"

	pathRemote     = "remote"
	pathLocal      = "local"
	pathAggregated = "aggregated"

	resultCompliant    = "compliant"
	resultTTFTMiss     = "ttft_miss"
	resultTPOTMiss     = "tpot_miss"
	resultTTFTTPOTMiss = "ttft_tpot_miss"
	resultFailed       = "failed"
)

var (
	_ requestcontrol.ResponseHeaderProcessor = &Plugin{}
	_ requestcontrol.ResponseBodyProcessor   = &Plugin{}
)

// Target is the latency target for one priority and, optionally, one prompt bucket.
type Target struct {
	Priority int `json:"priority"`
	// PromptBucket names a bucket from the plugin's bucket labels, e.g. "le1024". Empty matches
	// every bucket.
	PromptBucket string `json:"promptBucket,omitempty"`
	// TTFTMs is the time-to-first-token target. 0 does not score TTFT.
	TTFTMs float64 `json:"ttftMs,omitempty"`
	// TPOTMs is the mean time-per-output-token target. 0 does not score TPOT.
	TPOTMs float64 `json:"tpotMs,omitempty"`
}

// Parameters is the plugin configuration.
type Parameters struct {
	// PromptBuckets are increasing upper bounds, in prompt tokens. A request is labeled
	// "le<bound>" for the first bound it does not exceed, else "gt<last bound>".
	PromptBuckets []int `json:"promptBuckets,omitempty"`
	// PrefillProfileName is the scheduling profile whose result marks a remote prefill.
	PrefillProfileName string `json:"prefillProfileName,omitempty"`
	// Targets are matched by priority, then by bucket; a bucket-specific target wins. Requests at
	// a priority with no target are not scored.
	Targets []Target `json:"targets"`
}

// Plugin records SLO attainment. See the package README.
type Plugin struct {
	typedName      plugin.TypedName
	buckets        []int
	bucketLabels   []string
	prefillProfile string
	targets        map[int]map[string]Target // priority -> bucket ("" for any) -> target

	statuses sync.Map // key: RequestID, value: HTTP status string
}

// PluginFactory builds the plugin from its parameters.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	params := Parameters{PrefillProfileName: defaultPrefillProfile}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PluginType, err)
		}
	}
	p, err := newPlugin(name, params)
	if err != nil {
		return nil, err
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
	}
	return p, nil
}

func newPlugin(name string, params Parameters) (*Plugin, error) {
	if !slices.IsSorted(params.PromptBuckets) || len(slices.Compact(slices.Clone(params.PromptBuckets))) != len(params.PromptBuckets) {
		return nil, fmt.Errorf("%s plugin %q: promptBuckets must strictly increase", PluginType, name)
	}
	if params.PrefillProfileName == "" {
		params.PrefillProfileName = defaultPrefillProfile
	}
	labels := make([]string, 0, len(params.PromptBuckets)+1)
	for _, b := range params.PromptBuckets {
		if b <= 0 {
			return nil, fmt.Errorf("%s plugin %q: promptBuckets must be > 0", PluginType, name)
		}
		labels = append(labels, "le"+strconv.Itoa(b))
	}
	if n := len(params.PromptBuckets); n > 0 {
		labels = append(labels, "gt"+strconv.Itoa(params.PromptBuckets[n-1]))
	} else {
		labels = append(labels, "all")
	}
	if len(params.Targets) == 0 {
		return nil, fmt.Errorf("%s plugin %q: at least one target is required", PluginType, name)
	}
	targets := map[int]map[string]Target{}
	for i, t := range params.Targets {
		if t.TTFTMs < 0 || t.TPOTMs < 0 {
			return nil, fmt.Errorf("%s plugin %q: targets[%d]: targets must be >= 0", PluginType, name, i)
		}
		if t.PromptBucket != "" && !slices.Contains(labels, t.PromptBucket) {
			return nil, fmt.Errorf("%s plugin %q: targets[%d]: unknown promptBucket %q, want one of %v", PluginType, name, i, t.PromptBucket, labels)
		}
		if targets[t.Priority] == nil {
			targets[t.Priority] = map[string]Target{}
		}
		if _, dup := targets[t.Priority][t.PromptBucket]; dup {
			return nil, fmt.Errorf("%s plugin %q: targets[%d]: duplicate target for priority %d bucket %q", PluginType, name, i, t.Priority, t.PromptBucket)
		}
		targets[t.Priority][t.PromptBucket] = t
	}
	return &Plugin{
		typedName:      plugin.TypedName{Type: PluginType, Name: name},
		buckets:        params.PromptBuckets,
		bucketLabels:   labels,
		prefillProfile: params.PrefillProfileName,
		targets:        targets,
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *Plugin) TypedName() plugin.TypedName { return p.typedName }

// ResponseHeader records the response status for classification at end of stream.
func (p *Plugin) ResponseHeader(_ context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil {
		return
	}
	status := response.Headers[":status"]
	if status == "" {
		status = response.Headers["status"]
	}
	p.statuses.Store(request.RequestID, status)
}

// ResponseBody classifies the request at end of stream.
func (p *Plugin) ResponseBody(_ context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	v, _ := p.statuses.LoadAndDelete(request.RequestID)
	status, _ := v.(string)

	priority := request.Objectives.Priority
	bucket := p.bucket(promptTokens(request, response))
	target, ok := p.target(priority, bucket, request)
	if !ok {
		return
	}
	path := p.path(request)
	prio := strconv.Itoa(priority)
	fairness := metricsutil.BoundFairnessID(request.FairnessID)

	success := (status == "" || status == "200") && response.TerminationCause == requestcontrol.TerminationCauseNatural
	if !success {
		sloRequests.WithLabelValues(request.TargetModel, fairness, prio, bucket, path, resultFailed).Inc()
		return
	}

	ttftOK, tpotOK := true, true
	if !response.FirstTokenTimestamp.IsZero() && !response.ReceivedTimestamp.IsZero() {
		ttft := response.FirstTokenTimestamp.Sub(response.ReceivedTimestamp)
		sloTTFT.WithLabelValues(prio, bucket, path).Observe(ttft.Seconds())
		if target.TTFTMs > 0 {
			ttftOK = ttft <= msDuration(target.TTFTMs)
		}
	}
	if tpot, ok := meanTPOT(response); ok {
		sloTPOT.WithLabelValues(prio, path).Observe(tpot.Seconds())
		if target.TPOTMs > 0 {
			tpotOK = tpot <= msDuration(target.TPOTMs)
		}
	}
	sloRequests.WithLabelValues(request.TargetModel, fairness, prio, bucket, path, classify(ttftOK, tpotOK)).Inc()
}

func classify(ttftOK, tpotOK bool) string {
	switch {
	case ttftOK && tpotOK:
		return resultCompliant
	case !ttftOK && !tpotOK:
		return resultTTFTTPOTMiss
	case !ttftOK:
		return resultTTFTMiss
	default:
		return resultTPOTMiss
	}
}

// meanTPOT is the per-request mean gap between output tokens. It needs a streamed response with
// at least two output tokens.
func meanTPOT(response *requestcontrol.Response) (time.Duration, bool) {
	n := response.Usage.CompletionTokens
	if !response.Streaming || n < 2 || response.FirstTokenTimestamp.IsZero() || response.CompleteTimestamp.IsZero() {
		return 0, false
	}
	return response.CompleteTimestamp.Sub(response.FirstTokenTimestamp) / time.Duration(n-1), true
}

func promptTokens(request *scheduling.InferenceRequest, response *requestcontrol.Response) int {
	if response.Usage.PromptTokens > 0 {
		return response.Usage.PromptTokens
	}
	if request.Body != nil && request.Body.TokenizedRequest != nil {
		return request.Body.TokenizedRequest.TokenCount()
	}
	return 0
}

func (p *Plugin) bucket(tokens int) string {
	for i, b := range p.buckets {
		if tokens <= b {
			return p.bucketLabels[i]
		}
	}
	return p.bucketLabels[len(p.bucketLabels)-1]
}

// target resolves the configured target for priority and bucket, tightened by any per-request
// SLO headers.
func (p *Plugin) target(priority int, bucket string, request *scheduling.InferenceRequest) (Target, bool) {
	byBucket, ok := p.targets[priority]
	if !ok {
		return Target{}, false
	}
	t, ok := byBucket[bucket]
	if !ok {
		if t, ok = byBucket[""]; !ok {
			return Target{}, false
		}
	}
	t.TTFTMs = tighter(t.TTFTMs, headerMs(request.Headers, metadata.TTFTSLOHeaderKey))
	t.TPOTMs = tighter(t.TPOTMs, headerMs(request.Headers, metadata.TPOTSLOHeaderKey))
	return t, true
}

func (p *Plugin) path(request *scheduling.InferenceRequest) string {
	if request.SchedulingResult == nil {
		return pathAggregated
	}
	res, ok := request.SchedulingResult.ProfileResults[p.prefillProfile]
	switch {
	case !ok:
		return pathAggregated
	case res != nil && len(res.TargetEndpoints) > 0:
		return pathRemote
	default:
		return pathLocal
	}
}

func headerMs(headers map[string]string, key string) float64 {
	v, ok := metadata.GetLowerCaseHeaderValue(headers, key)
	if !ok {
		return 0
	}
	ms, err := strconv.ParseFloat(v, 64)
	if err != nil || ms <= 0 {
		return 0
	}
	return ms
}

// tighter returns the smaller positive target, treating 0 as unset.
func tighter(configured, requested float64) float64 {
	if requested > 0 && (configured == 0 || requested < configured) {
		return requested
	}
	return configured
}

func msDuration(ms float64) time.Duration {
	return time.Duration(ms * float64(time.Millisecond))
}

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{sloRequests, sloTTFT, sloTPOT} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register slo attainment metric: %w", err)
		}
	}
	return nil
}
