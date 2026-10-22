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

package sloattainment

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const testConfig = `{
	"promptBuckets": [1024, 8192],
	"targets": [
		{"priority": 100, "ttftMs": 500, "tpotMs": 30},
		{"priority": 100, "promptBucket": "gt8192", "ttftMs": 4000, "tpotMs": 30}
	]
}`

func newTestPlugin(t *testing.T) *Plugin {
	t.Helper()
	got, err := PluginFactory("slo", fwkplugin.StrictDecoder(json.RawMessage(testConfig)), nil)
	require.NoError(t, err)
	return got.(*Plugin)
}

type reqSpec struct {
	priority   int
	fairness   string
	prompt     int
	completion int
	ttft       time.Duration
	decode     time.Duration
	streaming  bool
	status     string
	cause      fwkrc.TerminationCause
	headers    map[string]string
	scheduling *fwksched.SchedulingResult
}

func run(p *Plugin, id string, s reqSpec) {
	headers := s.headers
	if headers == nil {
		headers = map[string]string{}
	}
	req := &fwksched.InferenceRequest{
		RequestID:        id,
		TargetModel:      "m",
		FairnessID:       s.fairness,
		Headers:          headers,
		Objectives:       fwksched.RequestObjectives{Priority: s.priority},
		SchedulingResult: s.scheduling,
	}
	status := s.status
	if status == "" {
		status = "200"
	}
	p.ResponseHeader(context.Background(), req, &fwkrc.Response{Headers: map[string]string{":status": status}}, nil)

	received := time.Unix(1000, 0)
	cause := s.cause
	if cause == "" {
		cause = fwkrc.TerminationCauseNatural
	}
	resp := &fwkrc.Response{
		EndOfStream:         true,
		TerminationCause:    cause,
		ReceivedTimestamp:   received,
		FirstTokenTimestamp: received.Add(s.ttft),
		CompleteTimestamp:   received.Add(s.ttft + s.decode),
		Streaming:           s.streaming,
	}
	resp.Usage.PromptTokens = s.prompt
	resp.Usage.CompletionTokens = s.completion
	p.ResponseBody(context.Background(), req, resp, nil)
}

func count(fairness, prio, bucket, path, result string) float64 {
	return promtestutil.ToFloat64(sloRequests.WithLabelValues("m", fairness, prio, bucket, path, result))
}

func TestFactory_InvalidConfig(t *testing.T) {
	cases := map[string]string{
		"no targets":       `{"promptBuckets":[1024]}`,
		"unsorted buckets": `{"promptBuckets":[8192,1024],"targets":[{"priority":1,"ttftMs":1}]}`,
		"dup buckets":      `{"promptBuckets":[1024,1024],"targets":[{"priority":1,"ttftMs":1}]}`,
		"unknown bucket":   `{"promptBuckets":[1024],"targets":[{"priority":1,"promptBucket":"le9","ttftMs":1}]}`,
		"negative target":  `{"targets":[{"priority":1,"ttftMs":-1}]}`,
		"duplicate target": `{"targets":[{"priority":1,"ttftMs":1},{"priority":1,"ttftMs":2}]}`,
		"unknown field":    `{"targets":[{"priority":1,"ttftMs":1}],"bogus":true}`,
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := PluginFactory("slo", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
			require.Error(t, err)
		})
	}
}

func TestBucketLabels(t *testing.T) {
	p := newTestPlugin(t)
	assert.Equal(t, "le1024", p.bucket(0))
	assert.Equal(t, "le1024", p.bucket(1024))
	assert.Equal(t, "le8192", p.bucket(1025))
	assert.Equal(t, "gt8192", p.bucket(8193))
}

func TestClassification(t *testing.T) {
	p := newTestPlugin(t)
	tests := []struct {
		name       string
		fairness   string
		spec       reqSpec
		bucket     string
		path       string
		wantResult string
	}{
		{
			name:       "fast stream is compliant",
			spec:       reqSpec{prompt: 100, completion: 101, ttft: 200 * time.Millisecond, decode: 2 * time.Second, streaming: true},
			bucket:     "le1024",
			path:       pathAggregated,
			wantResult: resultCompliant,
		},
		{
			name:       "slow first token",
			spec:       reqSpec{prompt: 100, completion: 101, ttft: 900 * time.Millisecond, decode: 2 * time.Second, streaming: true},
			bucket:     "le1024",
			path:       pathAggregated,
			wantResult: resultTTFTMiss,
		},
		{
			name:       "slow decode",
			spec:       reqSpec{prompt: 100, completion: 101, ttft: 200 * time.Millisecond, decode: 5 * time.Second, streaming: true},
			bucket:     "le1024",
			path:       pathAggregated,
			wantResult: resultTPOTMiss,
		},
		{
			name:       "both slow",
			spec:       reqSpec{prompt: 100, completion: 101, ttft: 900 * time.Millisecond, decode: 5 * time.Second, streaming: true},
			bucket:     "le1024",
			path:       pathAggregated,
			wantResult: resultTTFTTPOTMiss,
		},
		{
			name:       "long prompt uses its bucket target",
			spec:       reqSpec{prompt: 20000, completion: 2, ttft: 3 * time.Second, streaming: true},
			bucket:     "gt8192",
			path:       pathAggregated,
			wantResult: resultCompliant,
		},
		{
			name:       "non-streaming response scores TTFT only",
			spec:       reqSpec{prompt: 100, completion: 500, ttft: 300 * time.Millisecond, decode: 30 * time.Second},
			bucket:     "le1024",
			path:       pathAggregated,
			wantResult: resultCompliant,
		},
		{
			name:       "server error is failed",
			spec:       reqSpec{prompt: 100, status: "503"},
			bucket:     "le1024",
			path:       pathAggregated,
			wantResult: resultFailed,
		},
		{
			name:       "client disconnect is failed",
			spec:       reqSpec{prompt: 100, cause: fwkrc.TerminationCauseClientDisconnect},
			bucket:     "le1024",
			path:       pathAggregated,
			wantResult: resultFailed,
		},
		{
			name:       "header SLO tightens the target",
			spec:       reqSpec{prompt: 100, ttft: 300 * time.Millisecond, headers: map[string]string{metadata.TTFTSLOHeaderKey: "250"}},
			bucket:     "le1024",
			path:       pathAggregated,
			wantResult: resultTTFTMiss,
		},
		{
			name:       "header SLO cannot loosen the target",
			spec:       reqSpec{prompt: 100, ttft: 600 * time.Millisecond, headers: map[string]string{metadata.TTFTSLOHeaderKey: "5000"}},
			bucket:     "le1024",
			path:       pathAggregated,
			wantResult: resultTTFTMiss,
		},
		{
			name: "remote prefill path",
			spec: reqSpec{prompt: 100, ttft: 100 * time.Millisecond, scheduling: &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode":  {TargetEndpoints: []fwksched.Endpoint{nil}},
				"prefill": {TargetEndpoints: []fwksched.Endpoint{nil}},
			}}},
			bucket:     "le1024",
			path:       pathRemote,
			wantResult: resultCompliant,
		},
		{
			name: "declined prefill is the local path",
			spec: reqSpec{prompt: 100, ttft: 100 * time.Millisecond, scheduling: &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode":  {TargetEndpoints: []fwksched.Endpoint{nil}},
				"prefill": nil,
			}}},
			bucket:     "le1024",
			path:       pathLocal,
			wantResult: resultCompliant,
		},
	}
	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fairness := "t" + string(rune('a'+i))
			tc.spec.priority = 100
			tc.spec.fairness = fairness
			before := count(fairness, "100", tc.bucket, tc.path, tc.wantResult)
			run(p, "req-"+fairness, tc.spec)
			assert.Equal(t, before+1, count(fairness, "100", tc.bucket, tc.path, tc.wantResult))
		})
	}
}

func TestUnscoredPriorityRecordsNothing(t *testing.T) {
	p := newTestPlugin(t)
	before := count("unscored", "10", "le1024", pathAggregated, resultCompliant)
	run(p, "req-unscored", reqSpec{priority: 10, fairness: "unscored", prompt: 100, ttft: time.Millisecond})
	assert.Equal(t, before, count("unscored", "10", "le1024", pathAggregated, resultCompliant))
	_, held := p.statuses.Load("req-unscored")
	assert.False(t, held, "per-request status is released at end of stream")
}
