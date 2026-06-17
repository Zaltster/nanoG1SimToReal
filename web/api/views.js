// Page-view counter for the nanoG1 demo — Upstash Redis via REST, zero deps.
// Same pattern as humanoid-terminal/api/views.ts, but dependency-free (plain
// fetch) so it works on the static WASM deploy.
//
// Activate by setting these on the Vercel project (g1-sub60-walk):
//   UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN
// Until then it returns {views:null} and the demo hides the line.
module.exports = async function handler(req, res) {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) return res.status(200).json({ views: null });

  const cmd = req.method === 'POST' ? 'incr' : 'get';   // POST counts a visit; GET just reads
  try {
    const r = await fetch(`${url}/${cmd}/nanog1_views`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const j = await r.json();
    return res.status(200).json({ views: Number(j.result) || 0 });
  } catch {
    return res.status(200).json({ views: null });
  }
};
