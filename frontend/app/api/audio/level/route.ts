export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const PI_AUDIO_LEVEL = process.env.PI_AUDIO_LEVEL || "http://raspberrypi.local:5000/audio/level";

export async function GET() {
  const upstream = await fetch(PI_AUDIO_LEVEL, { cache: "no-store" });
  if (!upstream.ok || !upstream.body) {
    return new Response("Upstream error", { status: 502 });
  }
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}