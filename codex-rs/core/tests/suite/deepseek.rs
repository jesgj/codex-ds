//! End-to-end tests for running DeepSeek models through the Responses API.
//!
//! DeepSeek's own API emits the full Responses event set, but some gateways
//! (notably the OpenCode Zen free tier) stream a "lite" Responses protocol:
//! `response.output_text.delta` / `response.function_call_arguments.delta`
//! plus `response.completed`, without any `response.output_item.added` /
//! `response.output_item.done` events. These tests verify that such streams
//! drive a complete codex turn, including tool execution and multi-turn
//! history.

use codex_model_provider_info::DEEPSEEK_PROVIDER_ID;
use codex_model_provider_info::create_deepseek_provider;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;

/// Configured DeepSeek provider pointed at a test mock server.
///
/// The built-in provider reads `DEEPSEEK_API_KEY` from the environment; tests
/// strip the env key so requests go out unauthenticated to the mock.
fn deepseek_provider_for(base_url: String) -> codex_model_provider_info::ModelProviderInfo {
    let mut provider = create_deepseek_provider();
    provider.base_url = Some(base_url);
    provider.env_key = None;
    provider
}

/// Lite "output_text.delta" event, as emitted by the OpenCode Zen gateway.
fn lite_text_delta(delta: &str) -> serde_json::Value {
    json!({ "type": "response.output_text.delta", "delta": delta })
}

/// Lite "output_item.added" for a function call with empty arguments.
fn lite_function_call_added(output_index: i64, call_id: &str, name: &str) -> serde_json::Value {
    json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "arguments": ""
        }
    })
}

/// Lite "function_call_arguments.delta" keyed by output index.
fn lite_function_call_args_delta(output_index: i64, delta: &str) -> serde_json::Value {
    json!({
        "type": "response.function_call_arguments.delta",
        "output_index": output_index,
        "delta": delta
    })
}

/// Lite "response.completed" without usage details.
fn lite_completed(id: &str) -> serde_json::Value {
    json!({ "type": "response.completed", "response": { "id": id } })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deepseek_lite_text_stream_completes_turn() {
    skip_if_no_network!();

    let server = wiremock::MockServer::start().await;
    let resp_mock = mount_sse_sequence(
        &server,
        vec![sse(vec![
            lite_text_delta("Hello"),
            lite_text_delta(" from"),
            lite_text_delta(" DeepSeek"),
            lite_completed("resp1"),
        ])],
    )
    .await;

    let mut builder = test_codex()
        .with_model("deepseek-v4-flash")
        .with_config(move |config| {
            let base_url = config.model_provider.base_url.clone();
            config.model_provider_id = DEEPSEEK_PROVIDER_ID.to_string();
            config.model_provider = deepseek_provider_for(base_url.unwrap_or_default());
        });
    let test = builder
        .build(&server)
        .await
        .expect("create new conversation");

    test.submit_text_turn("hello").await.expect("submit turn");

    assert_eq!(resp_mock.requests().len(), 1);
    let request = resp_mock.requests().into_iter().next().expect("one request");
    assert_eq!(request.path(), "/v1/responses");
    assert_eq!(request.body_json()["model"], "deepseek-v4-flash");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deepseek_lite_stream_executes_tool_call_and_keeps_history() {
    skip_if_no_network!();

    let server = wiremock::MockServer::start().await;
    // Turn 1: lite stream with a shell tool call (no output_item.done events).
    // Turn 2: lite text stream after the tool output is fed back.
    let resp_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                lite_text_delta("Let me check"),
                lite_function_call_added(0, "call_1", "shell"),
                lite_function_call_args_delta(
                    0,
                    "{\"command\": \"echo hello\", \"cwd\": \"/tmp\", \"timeout\": 1000}",
                ),
                lite_completed("resp1"),
            ]),
            sse(vec![lite_text_delta("Done"), lite_completed("resp2")]),
        ],
    )
    .await;

    let mut builder = test_codex()
        .with_model("deepseek-v4-flash")
        .with_config(move |config| {
            let base_url = config.model_provider.base_url.clone();
            config.model_provider_id = DEEPSEEK_PROVIDER_ID.to_string();
            config.model_provider = deepseek_provider_for(base_url.unwrap_or_default());
        });
    let test = builder
        .build(&server)
        .await
        .expect("create new conversation");

    test.submit_text_turn("run a command").await.expect("submit turn");

    let requests = resp_mock.requests();
    assert_eq!(requests.len(), 2, "tool call should trigger a follow-up request");

    // The synthesized function_call item must have carried the accumulated
    // arguments, so the shell tool executed and its output came back in the
    // second request.
    let second = requests[1].clone();
    let call_output = second.function_call_output("call_1");
    assert_eq!(
        call_output.get("output").and_then(|v| v.as_array()).map(Vec::len),
        Some(2)
    );

    // The synthesized message item must be part of the follow-up input, so
    // the model sees its own prior text.
    let input_text = second.body_json()["input"]
        .as_array()
        .expect("input should be an array")
        .iter()
        .filter_map(|item| item.get("content"))
        .filter_map(|content| content.as_array())
        .flatten()
        .filter_map(|part| part.get("text"))
        .filter_map(|text| text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        input_text.contains("Let me check"),
        "follow-up input should contain the assistant text, got: {input_text:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deepseek_lite_reasoning_stream_completes_turn() {
    skip_if_no_network!();

    let server = wiremock::MockServer::start().await;
    let resp_mock = mount_sse_sequence(
        &server,
        vec![sse(vec![
            json!({"type": "response.reasoning_text.delta", "content_index": 0, "delta": "thinking"}),
            lite_text_delta("Answer"),
            lite_completed("resp1"),
        ])],
    )
    .await;

    let mut builder = test_codex()
        .with_model("deepseek-v4-flash")
        .with_config(move |config| {
            let base_url = config.model_provider.base_url.clone();
            config.model_provider_id = DEEPSEEK_PROVIDER_ID.to_string();
            config.model_provider = deepseek_provider_for(base_url.unwrap_or_default());
        });
    let test = builder
        .build(&server)
        .await
        .expect("create new conversation");

    test.submit_text_turn("hello").await.expect("submit turn");

    assert_eq!(resp_mock.requests().len(), 1);
}
