// Optional: a small Cloudflare Worker that lets the page give live generative answers
// without putting an API key in the page. The key stays here, as a secret.
//
// Set these in the Worker settings:
//   GEMINI_API_KEY  (secret)  a key from Google AI Studio
//   ALLOW_ORIGIN    (variable) https://YOUR-USERNAME.github.io
//   GEMINI_MODEL    (variable, optional) defaults to gemini-flash-latest
const SYS = "You answer questions inside a loan-model fairness audit built for a high school hackathon. " +
  "Use only the DATA and NOTES in the message. Quote exact numbers. If the answer is not there, say so. " +
  "Use under 120 words, plain text, no legal or financial advice.";

export default {
  async fetch(req, env) {
    const cors = {
      "Access-Control-Allow-Origin": env.ALLOW_ORIGIN || "*",
      "Access-Control-Allow-Headers": "content-type",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
    };
    const json = (body, status = 200) =>
      new Response(JSON.stringify(body), { status, headers: { ...cors, "content-type": "application/json" } });

    if (req.method === "OPTIONS") return new Response(null, { headers: cors });
    if (req.method !== "POST") return json({ error: "POST only" }, 405);

    let body;
    try { body = await req.json(); } catch { return json({ error: "bad json" }, 400); }
    const { question, context } = body;
    if (!question || question.length > 500 || !context || context.length > 40000) {
      return json({ error: "bad request" }, 400);
    }

    const model = env.GEMINI_MODEL || "gemini-flash-latest";
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", "x-goog-api-key": env.GEMINI_API_KEY },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYS }] },
        contents: [{ role: "user", parts: [{ text: `${context}\n\nQuestion: ${question}` }] }],
      }),
    });
    const j = await res.json();
    const text = j.candidates?.[0]?.content?.parts?.[0]?.text;
    return json(text ? { text } : { error: j.error?.message || "no answer" });
  },
};
