// app/api/camera/route.ts
export const runtime = "nodejs";            // NOT edge (needs long-lived TCP)
export const dynamic = "force-dynamic";     // don't cache the route

const PI_STREAM =
  process.env.PI_STREAM || "http://raspberrypi.local:5000/stream";

export async function GET() { 
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 5000);

  try {
    const upstream = await fetch(PI_STREAM, {
      signal: controller.signal,
      // don't cache a live stream
      cache: "no-store",
      // keep-alive helps some proxies
      headers: { Connection: "keep-alive" },
    });
    clearTimeout(t);

    if (!upstream.ok || !upstream.body) {
      return new Response("Upstream stream error", { status: 502 });
    }

    // Forward the MJPEG stream body + content-type boundary
    const contentType =
      upstream.headers.get("content-type") ||
      "multipart/x-mixed-replace; boundary=frame";

    return new Response(upstream.body, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "no-cache, no-store, must-revalidate",
        Pragma: "no-cache",
        Expires: "0",
        // CORS if you want to hit it from another origin
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch (err) {
    clearTimeout(t);
    const msg =
      err instanceof Error && err.name === "AbortError"
        ? "Timed out connecting to Pi stream"
        : "Failed to connect to Pi stream";
    return new Response(msg, { status: 504 });
  }
}