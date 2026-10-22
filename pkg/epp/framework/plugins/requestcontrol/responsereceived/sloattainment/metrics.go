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
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var (
	sloRequests = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "slo_requests_total",
			Help:      metricsutil.HelpMsgWithStability("Scored requests by SLO result: compliant, ttft_miss, tpot_miss, ttft_tpot_miss or failed.", compbasemetrics.ALPHA),
		},
		[]string{"target_model_name", "fairness_id", "priority", "prompt_bucket", "path", "result"},
	)

	sloTTFT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "slo_ttft_seconds",
			Help:      metricsutil.HelpMsgWithStability("Time to first token of scored successful requests, by priority, prompt bucket and path.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3, 4, 6, 8, 10, 15, 30, 60,
			},
		},
		[]string{"priority", "prompt_bucket", "path"},
	)

	sloTPOT = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "slo_tpot_seconds",
			Help:      metricsutil.HelpMsgWithStability("Mean time per output token of scored successful streaming requests, by priority and path.", compbasemetrics.ALPHA),
			Buckets: []float64{
				0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1,
			},
		},
		[]string{"priority", "path"},
	)
)
