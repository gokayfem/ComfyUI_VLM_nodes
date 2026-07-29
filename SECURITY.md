# API credential security

## Guarantees

- API keys are never accepted as node inputs, widget values, workflow fields,
  outputs, metadata, or log messages.
- Each built-in provider reads only its standard server-side environment
  variable and sends it only to that provider's fixed official HTTPS endpoint.
- A built-in provider key cannot be combined with a workflow-supplied URL.
- The custom endpoint reads only `CUSTOM_API_KEY`. Remote custom endpoints must
  use HTTPS; unencrypted and keyless requests are limited to loopback.
- HTTP redirects and environment proxies are disabled by default. Proxy use is
  an explicit non-secret node option for installations that require it.
- Hosted calls are stateless. No Python node-instance conversation history is
  retained, and OpenAI Responses requests set `store=false`.
- Exceptions are bounded and redact the resolved key, URL-encoded variants,
  bearer tokens, common provider-key formats, authorization fields, and URL
  user-info before the message reaches ComfyUI.
- Local image/video-frame uploads are uniformly sampled, resized,
  JPEG-compressed, limited to 4 MiB per image, and limited to 24 MiB total.
- User JSON Schemas are size/depth/node bounded and may contain only local
  fragment `$ref` values. Remote URLs and file references are rejected before
  validation, preventing schema resolution from becoming an SSRF or local-file
  access path.

## Configure credentials

Set the matching variable in the environment that launches ComfyUI, then
restart ComfyUI:

| Provider | Variable |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| xAI | `XAI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Custom remote endpoint | `CUSTOM_API_KEY` |

For an interactive POSIX/WSL session, this avoids putting the value in shell
history:

```bash
read -rsp "Provider API key: " OPENAI_API_KEY
export OPENAI_API_KEY
python main.py
```

Use the equivalent secret manager or service environment mechanism for a
persistent installation. Do not commit a `.env` file, workflow containing an
old key, shell script containing a key, or copied ComfyUI log.

Web search is disabled by default. Enabling it sends the request content to the
selected provider's server-side search system and may have separate retention,
regional-availability, and billing terms. Treat it as an explicit data-sharing
choice; do not enable it for content that is outside those terms.

## Legacy workflows

Versions before this security update exposed an `api_key` text widget.
The frontend migration clears position 3 of every serialized
`PromptGenerateAPI` node before LiteGraph creates the active node, including
nodes inside saved subgraph definitions. The backend independently rejects any
value that is not one of the two safe credential-source choices.

The source workflow file is not rewritten merely by opening it. Save the
migrated workflow, securely remove old copies, and rotate any credential that
was ever saved, shared, committed, backed up, or placed in an exported PNG.

## Threat boundary

ComfyUI custom nodes execute Python code with the permissions of the ComfyUI
process. Another untrusted custom-node package can read the same process
environment regardless of protections in this repository. Install only trusted
node packs, keep ComfyUI authenticated and bound to a trusted interface, and do
not expose an unauthenticated server to the public internet.

If a key may have been exposed, revoke it with the provider immediately, review
usage, create a replacement with the minimum needed project permissions and
spend limit, and restart ComfyUI with the replacement.
