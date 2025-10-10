// app/api/status/route.ts
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const PI_STATUS =
  process.env.PI_STATUS || "http://raspberrypi.local:5000/status";

type PiStatus = { unlocked: boolean; state?: string };

export async function GET() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000); // 5s

  try {
    const resp = await fetch(PI_STATUS, {
      signal: controller.signal,
      cache: "no-store",
      // Don't set Content-Type on a GET; let server decide
    });
    clearTimeout(timeout);

    if (!resp.ok) {
      return Response.json(
        { error: `Failed to fetch status: ${resp.status}` },
        { status: resp.status }
      );
    }

    // Try JSON first
    const ctype = resp.headers.get("content-type") || "";
    let out: PiStatus;

    if (ctype.includes("application/json")) {
      const data = (await resp.json()) as PiStatus;
      // Normalize/validate
      out = {
        unlocked: Boolean(data.unlocked),
        state: data.state ?? (data.unlocked ? "UNLOCKED" : "LOCKED"),
      };
    } else {
      // Fallback: plain text "LOCKED" / "UNLOCKED"
      const txt = (await resp.text()).trim().toUpperCase();
      const unlocked = txt.includes("UNLOCKED") || txt === "1" || txt === "TRUE";
      out = { unlocked, state: unlocked ? "UNLOCKED" : "LOCKED" };
    }

    return Response.json(out, {
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        Pragma: "no-cache",
        Expires: "0",
      },
    });
  } catch (err: any) {
    clearTimeout(timeout);
    const msg = err?.name === "AbortError" ? "Timed out" : (err?.message || "Unknown error");
    return Response.json({ error: msg }, { status: 504 });
  }
}