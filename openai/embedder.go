package openai

import (
	"context"
	"fmt"
	"math"

	"github.com/datar-psa/goeval/api"
)

// Embedder implements api.Embedder on the /embeddings endpoint
type Embedder struct {
	client
}

// NewEmbedder creates an embedder for an OpenAI-compatible server
// model: the embedding model id (e.g., "text-embedding-3-small")
func NewEmbedder(model string, opts ...Option) *Embedder {
	return &Embedder{client: newClient(model, opts)}
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Embed implements Embedder.Embed. The vector is normalized to length 1.
func (e *Embedder) Embed(ctx context.Context, text string) ([]float64, error) {
	body := map[string]any{
		"model":           e.model,
		"input":           text,
		"encoding_format": "float",
	}

	var resp embeddingResponse
	if err := e.post(ctx, "/embeddings", body, &resp); err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}
	if resp.Error != nil {
		return nil, fmt.Errorf("failed to generate embedding: provider error: %s", e.trim([]byte(resp.Error.Message)))
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	vec := resp.Data[0].Embedding
	if len(vec) == 0 {
		return nil, fmt.Errorf("empty embedding vector")
	}

	var sum float64
	for _, v := range vec {
		sum += v * v
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return nil, fmt.Errorf("zero embedding vector")
	}
	out := make([]float64, len(vec))
	for i, v := range vec {
		out[i] = v / norm
	}
	return out, nil
}

// Verify that Embedder implements Embedder
var _ api.Embedder = (*Embedder)(nil)
