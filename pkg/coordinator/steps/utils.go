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

package steps

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"

	"github.com/go-logr/logr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// maxErrorBodySize caps how much of a non-2xx upstream response body is read
// into memory, bounding OOM exposure to an adversarial upstream pod.
const maxErrorBodySize = 8 << 10 // 8 KB

// readErrorBody reads up to maxErrorBodySize of an upstream error response body.
func readErrorBody(r io.Reader) []byte {
	body, _ := io.ReadAll(io.LimitReader(r, maxErrorBodySize))
	return body
}

// upstreamError builds a pipeline.UpstreamError tagged with the step name so the
// server can map an upstream 4xx to a client error and a 5xx to a gateway fault.
func upstreamError(step string, statusCode int, body []byte) error {
	return &pipeline.UpstreamError{Step: step, StatusCode: statusCode, Body: string(body)}
}

// parseUseOpenAIFormat reads the use_openai_format step parameter, defaulting to
// true when absent. A present but non-bool value is a configuration error.
func parseUseOpenAIFormat(params map[string]any) (bool, error) {
	v, ok, err := paramBool(params, "use_openai_format")
	if err != nil {
		return false, err
	}
	if !ok {
		return true, nil
	}
	return v, nil
}

// rejectUseOpenAIFormatOverride returns an error if params sets use_openai_format.
// decode and conditional-decode derive their body format directly from the
// request's original path, so a step-level override has no effect; rejecting
// the key surfaces stale config instead of silently ignoring it.
func rejectUseOpenAIFormatOverride(step string, params map[string]any) error {
	if _, ok := params["use_openai_format"]; ok {
		return fmt.Errorf("%s: use_openai_format is not a valid parameter for this step", step)
	}
	return nil
}

// unreachableFormatError builds an error for a request format with no
// registered coordinator route (see server.go), signaling a routing bug
// rather than a client error.
func unreachableFormatError(format reqcommon.APIType) error {
	return fmt.Errorf("unsupported request format %v: no coordinator route serves it", format)
}

// resolveFormat maps a request path to the wire format a step emits. The steps
// build only Completions, Chat Completions, and generate bodies, so any other
// API collapses to APITypeVLLMGenerate; Chat Completions additionally requires
// useOpenAIFormat. Generate is the fallback because its body carries the prompt
// as reqCtx.TokenIDs and does not depend on the client's request shape.
func resolveFormat(useOpenAIFormat bool, path string) reqcommon.APIType {
	switch detected := reqcommon.DetectAPIType(path); detected {
	case reqcommon.APITypeCompletions:
		return detected
	case reqcommon.APITypeChatCompletions:
		if useOpenAIFormat {
			return detected
		}
	}
	return reqcommon.APITypeVLLMGenerate
}

// buildMMFeatures builds the multimodal features map (mm_hashes, mm_placeholders,
// and optionally kwargs_data) from the request's multimodal entries. It returns
// nil when there are no entries. Entries are grouped by Modality, so a
// mixed-modality request has one key per modality in each feature map.
func buildMMFeatures(entries []pipeline.MultimodalEntry, includeKwargs bool) map[string]any {
	if len(entries) == 0 {
		return nil
	}
	hashesByMod := make(map[string][]string)
	placeholdersByMod := make(map[string][]any)
	// Left nil unless the caller asked for kwargs_data: the decode and
	// conditional-decode bodies never carry it, and building it there would
	// allocate a map, a slice per modality, and a box per entry for nothing.
	var kwargsByMod map[string][]any
	if includeKwargs {
		kwargsByMod = make(map[string][]any)
	}
	for _, entry := range entries {
		mod := entry.Modality
		hashesByMod[mod] = append(hashesByMod[mod], entry.Hash)
		placeholdersByMod[mod] = append(placeholdersByMod[mod], map[string]any{
			"offset": entry.Placeholder.Offset,
			"length": entry.Placeholder.Length,
		})
		if includeKwargs {
			kwargsByMod[mod] = append(kwargsByMod[mod], kwargsSentinel(entry.KwargsData))
		}
	}
	features := map[string]any{
		"mm_hashes":       hashesByMod,
		"mm_placeholders": placeholdersByMod,
	}
	if includeKwargs {
		features["kwargs_data"] = kwargsByMod
	}
	return features
}

// validateEntryModalities rejects a MultimodalEntry with no Modality; steps
// that read entries call it first. Both producers set the field, so reaching
// here means a coordinator bug: the error is deliberately not ErrBadRequest,
// since such a request should surface as a 5xx rather than blame the caller.
//
// Failing is what keeps a mislabeled entry from corrupting a response. Every
// reader pairs per-modality by position on this field, so an entry defaulted to
// some modality would splice into that modality's index sequence, pair with
// another entry's part or response slot, and shift every later entry sharing
// the label. The request would complete on a guess.
func validateEntryModalities(entries []pipeline.MultimodalEntry) error {
	for i, entry := range entries {
		if entry.Modality == "" {
			return fmt.Errorf("multimodal entry %d has no modality (hash %q)", i, entry.Hash)
		}
	}
	return nil
}

// kwargsSentinel implements the JSON-null "resolve from cache" convention for
// one kwargs_data slot. Our sentinel is the empty string, which MUST serialize
// as null, not "": vLLM reads null (or an absent field) as a cache-hit item to
// fetch by hash, while "" is decoded as an inline tensor and fails with "Input
// data was truncated". Non-empty entries are base64 tensor blobs, verbatim.
func kwargsSentinel(k string) any {
	if k == "" {
		return nil
	}
	return k
}

// singleEntryKwargs builds a kwargs_data value for one entry: each encode
// fanout sub-request carries exactly one entry's kwargs under its modality
// key. Used by encode.buildEncodeBody.
func singleEntryKwargs(modality, kwargs string) map[string][]any {
	return map[string][]any{modality: {kwargsSentinel(kwargs)}}
}

// coerceParamsMap coerces a transfer-params value from an upstream response to a
// map: a non-object value is logged at debug and skipped (returns nil) rather
// than failing the request. A missing or null value is already nil; an empty map
// passes through so the connector's own no-metadata handling applies. label
// names the field for the debug log (e.g. "kv_transfer_params").
func coerceParamsMap(logger logr.Logger, v any, label string) map[string]any {
	switch m := v.(type) {
	case nil:
		return nil
	case map[string]any:
		return m
	default:
		logger.V(logutil.DEBUG).Info(label+" is not a JSON object; skipping",
			"type", fmt.Sprintf("%T", v))
		return nil
	}
}

// toIntSlice converts a JSON-unmarshalled []any of numeric elements to []int.
// Each element must be a non-negative integer represented as float64 or json.Number.
// The returned error identifies the offending element by index and wraps
// pipeline.ErrBadRequest.
func toIntSlice(values []any) ([]int, error) {
	out := make([]int, 0, len(values))
	for i, v := range values {
		n, err := anyToNonNegativeInt(v)
		if err != nil {
			return nil, fmt.Errorf("invalid token at index %d: %v: %w", i, err, pipeline.ErrBadRequest)
		}
		out = append(out, n)
	}
	return out, nil
}

// anyToNonNegativeInt converts a single JSON-unmarshalled numeric value to a non-negative int.
func anyToNonNegativeInt(v any) (int, error) {
	switch n := v.(type) {
	case float64:
		if n < 0 || n != math.Trunc(n) {
			return 0, fmt.Errorf("expected non-negative integer, got %v", v)
		}
		// An in-range integer-valued float64 round-trips through int; a value
		// too large to fit does not (the conversion saturates), so this rejects
		// overflow without depending on the fragile float64(MaxInt) boundary.
		i := int(n)
		if float64(i) != n {
			return 0, fmt.Errorf("expected non-negative integer, got %v", v)
		}
		return i, nil
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0, err
		}
		if i < 0 || i > math.MaxInt {
			return 0, fmt.Errorf("expected non-negative integer, got %d", i)
		}
		return int(i), nil
	default:
		return 0, fmt.Errorf("expected number, got %T", v)
	}
}

// extractTokenIDs converts body["token_ids"] from a JSON-unmarshalled value to []int.
// Returns ErrBadRequest when the field is absent, not an array, empty, or contains
// non-integer or negative values.
func extractTokenIDs(raw any) ([]int, error) {
	if raw == nil {
		return nil, fmt.Errorf("token_ids is required: %w", pipeline.ErrBadRequest)
	}
	arr, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("token_ids must be an array, got %T: %w", raw, pipeline.ErrBadRequest)
	}
	if len(arr) == 0 {
		return nil, fmt.Errorf("token_ids must not be empty: %w", pipeline.ErrBadRequest)
	}
	return toIntSlice(arr)
}

// mmModalityArray reads features[field][modality] as a JSON array. present is
// false when field or its per-modality entry is absent or null, a valid "no
// such modality" state rather than an error. A present value of the wrong type
// is ErrBadRequest, so a malformed request fails loudly instead of reading as
// absent.
func mmModalityArray(features map[string]any, field, modality string) (arr []any, present bool, err error) {
	rawField, ok := features[field]
	if !ok || rawField == nil {
		return nil, false, nil
	}
	m, ok := rawField.(map[string]any)
	if !ok {
		return nil, false, fmt.Errorf("%s must be an object: %w", field, pipeline.ErrBadRequest)
	}
	raw, ok := m[modality]
	if !ok || raw == nil {
		return nil, false, nil
	}
	arr, ok = raw.([]any)
	if !ok {
		return nil, false, fmt.Errorf("%s[%s] must be an array: %w", field, modality, pipeline.ErrBadRequest)
	}
	return arr, true, nil
}

// modalitiesInFeatures returns the modality keys present in features[field],
// sorted so entry ordering stays deterministic across map iteration for tests
// and stable-order consumers. (nil, nil) when absent or an empty object,
// ErrBadRequest when present but not an object. An empty key is rejected here
// because this is where a client-supplied features map becomes entries, and the
// key becomes MultimodalEntry.Modality, every reader's key for positional
// pairing; validateEntryModalities states what an untagged entry would cost.
func modalitiesInFeatures(features map[string]any, field string) ([]string, error) {
	raw, ok := features[field]
	if !ok || raw == nil {
		return nil, nil
	}
	m, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an object: %w", field, pipeline.ErrBadRequest)
	}
	var out []string
	for k, v := range m {
		if v == nil {
			continue
		}
		if k == "" {
			return nil, fmt.Errorf("%s has an empty modality key: %w", field, pipeline.ErrBadRequest)
		}
		out = append(out, k)
	}
	sort.Strings(out)
	return out, nil
}

// checkModalitiesHashed reports an error when mm_placeholders or kwargs_data
// carries a modality hashed does not name: only modalities mm_hashes names
// become entries, so skipping the extra key would drop the item from the
// prefill and decode bodies while its placeholder tokens stay in token_ids,
// leaving the engine placeholders with nothing behind them.
func checkModalitiesHashed(features map[string]any, hashed []string) error {
	known := make(map[string]struct{}, len(hashed))
	for _, mod := range hashed {
		known[mod] = struct{}{}
	}
	for _, field := range []string{"mm_placeholders", "kwargs_data"} {
		mods, err := modalitiesInFeatures(features, field)
		if err != nil {
			return err
		}
		for _, mod := range mods {
			if _, ok := known[mod]; !ok {
				return fmt.Errorf("%s[%s] has no matching mm_hashes[%s]: %w",
					field, mod, mod, pipeline.ErrBadRequest)
			}
		}
	}
	return nil
}

// extractMultimodalEntries builds []pipeline.MultimodalEntry from the parallel
// slices in a generate-format features map. Each modality key under mm_hashes
// produces a run of entries, modalities visited in sorted order for
// determinism. Returns nil for a text-only request.
//
// mm_hashes names the modality set, so a modality only the other fields carry
// is ErrBadRequest: it describes an item with no hash to build an entry from.
// Per modality, mm_hashes and mm_placeholders are required and of equal length;
// kwargs_data is optional, and an absent field or a null item means "resolve
// from the encoder cache by hash", which maps to an empty KwargsData. A wrong
// type, a length mismatch, or an unexpected element is ErrBadRequest.
func extractMultimodalEntries(features map[string]any) ([]pipeline.MultimodalEntry, error) {
	if features == nil {
		return nil, nil
	}
	modalities, err := modalitiesInFeatures(features, "mm_hashes")
	if err != nil {
		return nil, err
	}
	if err := checkModalitiesHashed(features, modalities); err != nil {
		return nil, err
	}
	if len(modalities) == 0 {
		return nil, nil
	}

	var entries []pipeline.MultimodalEntry
	for _, mod := range modalities {
		rawHashes, _, err := mmModalityArray(features, "mm_hashes", mod)
		if err != nil {
			return nil, err
		}

		rawPlaceholders, present, err := mmModalityArray(features, "mm_placeholders", mod)
		if err != nil {
			return nil, err
		}

		rawKwargs, hasKwargs, err := mmModalityArray(features, "kwargs_data", mod)
		if err != nil {
			return nil, err
		}

		n := len(rawHashes)
		if !present && n > 0 {
			return nil, fmt.Errorf("mm_placeholders[%s] is required when mm_hashes[%s] is set: %w",
				mod, mod, pipeline.ErrBadRequest)
		}
		if len(rawPlaceholders) != n {
			return nil, fmt.Errorf("features length mismatch for %s: mm_hashes has %d, mm_placeholders has %d: %w",
				mod, n, len(rawPlaceholders), pipeline.ErrBadRequest)
		}
		// When present, kwargs_data is parallel to mm_hashes: full length with
		// nulls for cached items, never shortened. Metadata-only (cache-hit)
		// requests omit the field entirely.
		if hasKwargs && len(rawKwargs) != n {
			return nil, fmt.Errorf("features length mismatch for %s: mm_hashes has %d, kwargs_data has %d: %w",
				mod, n, len(rawKwargs), pipeline.ErrBadRequest)
		}

		for i := 0; i < n; i++ {
			hash, ok := rawHashes[i].(string)
			if !ok {
				return nil, fmt.Errorf("mm_hashes[%s][%d] must be a string: %w", mod, i, pipeline.ErrBadRequest)
			}

			pMap, ok := rawPlaceholders[i].(map[string]any)
			if !ok {
				return nil, fmt.Errorf("mm_placeholders[%s][%d] must be an object: %w", mod, i, pipeline.ErrBadRequest)
			}
			// The non-negative guarantee is load-bearing:
			// EncodeStep.buildEncodeTokenIDs indexes fullTokenIDs[offset]
			// (upper-bound guarded only) and allocates make([]int, 1+length),
			// which panics on a negative. vLLM accepts negatives here, so this
			// is deliberately stricter; do not relax it to a plain int parse.
			offset, err := anyToNonNegativeInt(pMap["offset"])
			if err != nil {
				return nil, fmt.Errorf("mm_placeholders[%s][%d].offset: %v: %w", mod, i, err, pipeline.ErrBadRequest)
			}
			length, err := anyToNonNegativeInt(pMap["length"])
			if err != nil {
				return nil, fmt.Errorf("mm_placeholders[%s][%d].length: %v: %w", mod, i, err, pipeline.ErrBadRequest)
			}

			// Empty KwargsData is the "resolve from cache" sentinel: either the
			// whole kwargs_data field is absent or this item is null.
			var kwarg string
			if hasKwargs {
				switch k := rawKwargs[i].(type) {
				case string:
					kwarg = k
				case nil:
				default:
					return nil, fmt.Errorf("kwargs_data[%s][%d] must be a string or null: %w", mod, i, pipeline.ErrBadRequest)
				}
			}

			entries = append(entries, pipeline.MultimodalEntry{
				Modality:   mod,
				Hash:       hash,
				KwargsData: kwarg,
				Placeholder: pipeline.PlaceholderRange{
					Offset: offset,
					Length: length,
				},
			})
		}
	}
	return entries, nil
}

// validatePlaceholderBounds checks that every placeholder span [offset,
// offset+length) lies within a prompt of tokenCount tokens. It guards the
// generate path, where the client supplies placeholder geometry directly:
// EncodeStep.buildEncodeTokenIDs indexes token_ids[offset] and allocates
// make([]int, 1+length), so an out-of-range offset reads the wrong token and an
// unbounded length (a tiny request can claim billions) is a memory-exhaustion
// vector. vLLM declares offset/length as plain unbounded ints on the generate
// endpoint and does not enforce this, so the coordinator does. offset and
// length are already guaranteed non-negative by extractMultimodalEntries.
func validatePlaceholderBounds(entries []pipeline.MultimodalEntry, tokenCount int) error {
	for i, e := range entries {
		off := e.Placeholder.Offset
		length := e.Placeholder.Length
		if off >= tokenCount {
			return fmt.Errorf("mm_placeholders[%d].offset %d out of range for %d token_ids: %w",
				i, off, tokenCount, pipeline.ErrBadRequest)
		}
		// off < tokenCount, so tokenCount-off is positive and cannot overflow.
		if length > tokenCount-off {
			return fmt.Errorf("mm_placeholders[%d] span (offset %d + length %d) exceeds %d token_ids: %w",
				i, off, length, tokenCount, pipeline.ErrBadRequest)
		}
	}
	return nil
}
