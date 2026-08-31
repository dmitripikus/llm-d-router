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

package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func TestExtractMMItems(t *testing.T) {
	tests := []struct {
		name     string
		request  map[string]any
		expected int
	}{
		{
			name: "no multimodal items",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "Hello, world!",
					},
				},
			},
			expected: 0,
		},
		{
			name: "single image item",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type": "text",
								"text": "What's in this image?",
							},
							map[string]any{
								"type": "image_url",
								"image_url": map[string]any{
									"url": "https://example.com/image.jpg",
								},
							},
						},
					},
				},
			},
			expected: 1,
		},
		{
			name: "multiple multimodal items",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type": "image_url",
								"image_url": map[string]any{
									"url": "https://example.com/image1.jpg",
								},
							},
							map[string]any{
								"type": "audio_url",
								"audio_url": map[string]any{
									"url": "https://example.com/audio.mp3",
								},
							},
							map[string]any{
								"type": "text",
								"text": "Describe these",
							},
						},
					},
				},
			},
			expected: 2,
		},
		{
			name: "input_audio type",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type": "input_audio",
								"input_audio": map[string]any{
									"data":   "base64data",
									"format": "wav",
								},
							},
						},
					},
				},
			},
			expected: 1,
		},
		{
			name: "single video item",
			request: map[string]any{
				"messages": []any{
					map[string]any{
						"role": "user",
						"content": []any{
							map[string]any{
								"type": "video_url",
								"video_url": map[string]any{
									"url": "https://example.com/video.mp4",
								},
							},
						},
					},
				},
			},
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			items := extractMMItems(tt.request)
			assert.Equal(t, tt.expected, len(items), "unexpected number of MM items")
		})
	}
}

func TestBuildEncoderRequest(t *testing.T) {
	originalRequest := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "text",
						"text": "What's in this image?",
					},
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url": "https://example.com/image.jpg",
						},
					},
				},
			},
		},
		"max_tokens": 100,
		"stream":     true,
	}

	mmItem := map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": "https://example.com/image.jpg",
		},
	}

	encoderRequest := buildEncoderRequest(originalRequest, mmItem)

	// Verify encoder request modifications
	assert.Equal(t, 1, encoderRequest["max_tokens"])
	assert.Equal(t, false, encoderRequest["stream"])
	_, hasStreamOptions := encoderRequest["stream_options"]
	assert.False(t, hasStreamOptions)

	// Verify messages contain only the MM item
	messages, ok := encoderRequest["messages"].([]map[string]any)
	assert.True(t, ok)
	assert.Equal(t, 1, len(messages))

	content, ok := messages[0]["content"].([]map[string]any)
	assert.True(t, ok)
	assert.Equal(t, 1, len(content))
	assert.Equal(t, "image_url", content[0]["type"])
}

// TestBuildEncoderRequest_MaxCompletionTokens is a regression test: a shallow
// copy previously left the client's max_completion_tokens value untouched
// alongside the newly-capped max_tokens=1, so a reasoning-model client's
// large max_completion_tokens would survive uncapped into the encoder request.
func TestBuildEncoderRequest_MaxCompletionTokens(t *testing.T) {
	originalRequest := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url": "https://example.com/image.jpg",
						},
					},
				},
			},
		},
		"max_tokens":            50,
		"max_completion_tokens": 100,
	}

	mmItem := map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": "https://example.com/image.jpg",
		},
	}

	encoderRequest := buildEncoderRequest(originalRequest, mmItem)

	assert.Equal(t, 1, encoderRequest["max_tokens"])
	assert.Equal(t, 1, encoderRequest["max_completion_tokens"])
}

func TestMMItemURL(t *testing.T) {
	tests := []struct {
		name     string
		item     map[string]any
		expected string
	}{
		{
			name: "image_url with url",
			item: map[string]any{
				"type": "image_url",
				"image_url": map[string]any{
					"url": "https://example.com/image.jpg",
				},
			},
			expected: "https://example.com/image.jpg",
		},
		{
			name: "audio_url with url",
			item: map[string]any{
				"type": "audio_url",
				"audio_url": map[string]any{
					"url": "https://example.com/audio.mp3",
				},
			},
			expected: "https://example.com/audio.mp3",
		},
		{
			name: "video_url with url",
			item: map[string]any{
				"type": "video_url",
				"video_url": map[string]any{
					"url": "https://example.com/video.mp4",
				},
			},
			expected: "https://example.com/video.mp4",
		},
		{
			name: "input_audio has no url",
			item: map[string]any{
				"type": "input_audio",
				"input_audio": map[string]any{
					"data":   "base64data",
					"format": "wav",
				},
			},
			expected: "",
		},
		{
			name:     "text type has no url",
			item:     map[string]any{"type": "text", "text": "hello"},
			expected: "",
		},
		{
			name: "image_url missing nested url field",
			item: map[string]any{
				"type":      "image_url",
				"image_url": map[string]any{},
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, mmItemURL(tt.item))
		})
	}
}

// imageURLItem builds an image_url content item.
func imageURLItem(url string) map[string]any {
	return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}
}

// videoURLItem builds a video_url content item.
func videoURLItem(url string) map[string]any {
	return map[string]any{"type": "video_url", "video_url": map[string]any{"url": url}}
}

// audioURLItem builds an audio_url content item.
func audioURLItem(url string) map[string]any {
	return map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": url}}
}

// inlineAudioItem builds an input_audio content item.
func inlineAudioItem(data, format string) map[string]any {
	return map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": data, "format": format}}
}

// userMessageRequest wraps content items in a minimal chat-completions request.
func userMessageRequest(items ...map[string]any) map[string]any {
	content := make([]any, len(items))
	for i, item := range items {
		content[i] = item
	}
	return map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": content},
		},
	}
}

func TestFanoutEncoderPrimerDeduplication(t *testing.T) {
	var requestCount atomic.Int32
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}]}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	encoderHostPort := encoderURL.Host

	tests := []struct {
		name          string
		request       map[string]any
		expectedCalls int32
	}{
		{
			name:          "no duplicates — all items sent",
			request:       userMessageRequest(imageURLItem("https://example.com/img1.jpg"), imageURLItem("https://example.com/img2.jpg")),
			expectedCalls: 2,
		},
		{
			name:          "duplicate image URLs — second is skipped",
			request:       userMessageRequest(imageURLItem("https://example.com/same.jpg"), imageURLItem("https://example.com/same.jpg")),
			expectedCalls: 1,
		},
		{
			name:          "duplicate video URLs — second is skipped",
			request:       userMessageRequest(videoURLItem("https://example.com/same.mp4"), videoURLItem("https://example.com/same.mp4")),
			expectedCalls: 1,
		},
		{
			name:          "inline audio items are never deduplicated",
			request:       userMessageRequest(inlineAudioItem("aaa", "wav"), inlineAudioItem("aaa", "wav")),
			expectedCalls: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requestCount.Store(0)
			err := srv.fanoutEncoderPrimer(context.Background(), tt.request, []string{encoderHostPort}, "test-req-id")
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedCalls, requestCount.Load())
		})
	}
}

// TestHandleECSharedStorage_Video verifies video_url items trigger the primer
// fire-and-forget flow: encoders are called once per distinct video URL, and
// the request body is passed through to the P/D connector unchanged. Unlike
// nixl, shared_storage discards encoder responses — no ec_transfer_params
// should appear on the outgoing prefill body.
func TestHandleECSharedStorage_Video(t *testing.T) {
	var encoderCalls atomic.Int32
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		encoderCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}]}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	var capturedBody []byte
	srv.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ string, _ APIType) {
		buf, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		capturedBody = buf
	}

	reqBody, _ := json.Marshal(userMessageRequest(
		videoURLItem("https://example.com/v1.mp4"),
		videoURLItem("https://example.com/v2.mp4"),
	))
	httpReq := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, io.NopCloser(bytes.NewReader(reqBody)))
	rw := httptest.NewRecorder()

	srv.handleECSharedStorage(rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host})

	assert.Equal(t, int32(2), encoderCalls.Load(), "encoder should be called once per distinct video URL")
	if !assert.NotNil(t, capturedBody, "handlePDConnector should have been invoked") {
		return
	}
	var parsed map[string]any
	assert.NoError(t, json.Unmarshal(capturedBody, &parsed))
	_, hasEC := parsed[requestFieldECTransferParams]
	assert.False(t, hasEC, "shared_storage primer must not add ec_transfer_params to the prefill body")
}

// TestHandleECSharedStorage_Audio verifies input_audio items trigger the
// primer once per item. Inline audio is never deduplicated, so two distinct
// input_audio blocks (even with the same data) produce two encoder calls.
func TestHandleECSharedStorage_Audio(t *testing.T) {
	var encoderCalls atomic.Int32
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		encoderCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}]}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	var capturedBody []byte
	srv.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ string, _ APIType) {
		buf, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		capturedBody = buf
	}

	reqBody, _ := json.Marshal(userMessageRequest(
		inlineAudioItem("aaa", "wav"),
		inlineAudioItem("bbb", "wav"),
	))
	httpReq := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, io.NopCloser(bytes.NewReader(reqBody)))
	rw := httptest.NewRecorder()

	srv.handleECSharedStorage(rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host})

	assert.Equal(t, int32(2), encoderCalls.Load(), "encoder should be called once per input_audio item")
	if !assert.NotNil(t, capturedBody, "handlePDConnector should have been invoked") {
		return
	}
	var parsed map[string]any
	assert.NoError(t, json.Unmarshal(capturedBody, &parsed))
	_, hasEC := parsed[requestFieldECTransferParams]
	assert.False(t, hasEC, "shared_storage primer must not add ec_transfer_params to the prefill body")
}
