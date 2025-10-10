"use client"

import { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Lock, LockOpen, Activity, RefreshCw } from "lucide-react"

export function SmartDoorDashboard() {
  const [doorLocked, setDoorLocked] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<string>("—")
  const pollingRef = useRef<NodeJS.Timeout | null>(null)

    // ---- Microphone state ----
  const AUDIO_SRC = process.env.NEXT_PUBLIC_AUDIO_SRC || "/api/audio/level" // SSE proxy to Pi
  const [dbfs, setDbfs] = useState<number>(-120)
  const [level, setLevel] = useState<number>(0) // 0..1 normalized
  const [micConnected, setMicConnected] = useState<boolean>(false)
  const esRef = useRef<EventSource | null>(null)

  async function fetchDeviceStates() {
    setIsLoading(true)
    setError(null)
    try {
      const res = await fetch("/api/status", {
        cache: "no-store",          // stop caching
        headers: { "Accept": "application/json, text/plain" },
      })

      // Try JSON first, then fallback to text
      let unlocked = false
      const contentType = res.headers.get("content-type") || ""
      if (contentType.includes("application/json")) {
        const data = await res.json()
        // expected: { unlocked: boolean, state?: "LOCKED"|"UNLOCKED" }
        unlocked = Boolean(data?.unlocked)
      } else {
        const txt = (await res.text()).trim().toUpperCase()
        unlocked = /UNLOCKED|ON|TRUE|1/.test(txt)
      }

      setDoorLocked(!unlocked)
      setLastUpdated(new Date().toLocaleTimeString())
    } catch (e: any) {
      setError(e?.message || "Failed to fetch device status")
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch immediately on mount and then poll every 2s
  useEffect(() => {
    fetchDeviceStates()
    pollingRef.current = setInterval(fetchDeviceStates, 2000)
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
    }
  }, [])

    // Connect to mic SSE on mount
  useEffect(() => {
    const es = new EventSource(AUDIO_SRC)
    esRef.current = es
    let alive = true

    es.onopen = () => setMicConnected(true)
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        if (!alive) return
        const newDb = typeof data.dbfs === "number" ? data.dbfs : -120
        const newLevel = typeof data.level === "number" ? data.level : 0
        setDbfs(Math.round(newDb * 10) / 10)
        setLevel(Math.max(0, Math.min(1, newLevel)))
      } catch {
        // ignore malformed chunks
      }
    }
    es.onerror = () => {
      setMicConnected(false)
      // EventSource auto-reconnects; keep it open
    }

    return () => {
      alive = false
      es.close()
    }
  }, [AUDIO_SRC])

  const levelPct = Math.round(level * 100)

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground">Smart Door Control</h1>
            <p className="text-sm text-muted-foreground">
              IoT Dashboard — Raspberry Pi (camera+ML) & ESP32 (servo)
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground">Last update: {lastUpdated}</span>
            <Button
              onClick={fetchDeviceStates}
              disabled={isLoading}
              variant="outline"
              size="sm"
              className="flex items-center gap-2"
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
              {isLoading ? "Refreshing..." : "Refresh"}
            </Button>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {/* Camera */}
          <Card className="border-border bg-card lg:col-span-2">
            <CardHeader>
              <div className="flex items-center gap-2">
                <CardTitle>Live Camera</CardTitle>
                <Badge variant="secondary">MediaPipe on Raspberry Pi</Badge>
              </div>
              <CardDescription>Annotated stream with hand detection overlay</CardDescription>
            </CardHeader>
            <CardContent>
              <img
                src="/api/camera"
                alt="Camera"
                className="w-full h-auto rounded-md border"
                style={{ background: "#000" }}
              />
            </CardContent>
          </Card>

          {/* Device Controls */}
          <Card className="border-border bg-card">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-primary" />
                <CardTitle>Door Lock</CardTitle>
              </div>
              <CardDescription>State reported by the system</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between rounded-md border bg-secondary/30 p-3">
                <div className="flex items-center gap-2">
                  {doorLocked ? (
                    <Lock className="h-5 w-5 text-red-500" />
                  ) : (
                    <LockOpen className="h-5 w-5 text-green-500" />
                  )}
                  <div>
                    <p className="font-medium text-foreground">Status</p>
                    <p className="text-xs text-muted-foreground">Auto-refreshed every 2s</p>
                  </div>
                </div>
                <Badge variant={doorLocked ? "destructive" : "default"}>
                  {doorLocked ? "Locked" : "Unlocked"}
                </Badge>
              </div>

              {error && (
                <p className="text-xs text-red-500">
                  {error}
                </p>
              )}

              <div className="mt-4 space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-foreground">Microphone</p>
                  <Badge variant={micConnected ? "default" : "destructive"}>
                    {micConnected ? "Streaming" : "Disconnected"}
                  </Badge>
                </div>

                <div className="text-xs text-muted-foreground">Level: {dbfs} dBFS</div>

                <div className="h-3 w-full overflow-hidden rounded bg-secondary">
                  <div
                    className="h-full bg-green-500 transition-[width] duration-75"
                    style={{ width: `${levelPct}%` }}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={levelPct}
                    role="progressbar"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}