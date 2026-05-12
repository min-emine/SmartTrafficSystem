"use client";

import type { CSSProperties, ReactNode } from "react";
import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bell,
  Camera,
  CirclePause,
  CirclePlay,
  Download,
  Gauge,
  GitBranch,
  LayoutDashboard,
  MapPinned,
  RefreshCw,
  Route,
  Settings,
  ShieldCheck,
  Siren,
  SlidersHorizontal,
  Timer,
  Zap
} from "lucide-react";
import {
  activityLog,
  lanes,
  manualZones,
  systemConfig,
  vehicleWeights,
  type Lane,
  type SignalState
} from "../data/traffic";

const feedTabs = [
  { id: "live", label: "Live", asset: "/traffic-live.gif" },
  { id: "learning", label: "Learning", asset: "/traffic-learning.gif" },
  { id: "zones", label: "Zones", asset: "/traffic-live.gif" }
] as const;

const navItems = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Signals", icon: Zap },
  { label: "Routes", icon: Route },
  { label: "Zones", icon: MapPinned },
  { label: "Settings", icon: Settings }
];

const signalTone: Record<SignalState, string> = {
  green: "signal-green",
  amber: "signal-amber",
  red: "signal-red"
};

function MetricCard({
  icon,
  label,
  value,
  detail,
  tone = "neutral"
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
  tone?: "neutral" | "good" | "warn" | "danger";
}) {
  return (
    <article className={`metric metric-${tone}`}>
      <div className="metric-icon" aria-hidden="true">
        {icon}
      </div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        <span>{detail}</span>
      </div>
    </article>
  );
}

function IconButton({
  label,
  children,
  active,
  onClick
}: {
  label: string;
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      className={`icon-button${active ? " is-active" : ""}`}
      type="button"
      aria-label={label}
      aria-pressed={active}
      title={label}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

export function TrafficDashboard() {
  const [mode, setMode] = useState<"autonomous" | "manual">("autonomous");
  const [isRunning, setIsRunning] = useState(true);
  const [selectedFeed, setSelectedFeed] = useState<(typeof feedTabs)[number]["id"]>("live");
  const [manualLane, setManualLane] = useState(0);
  const [greenSeconds, setGreenSeconds] = useState(42);
  const [learningFrames, setLearningFrames] = useState(systemConfig.learningFrames);
  const [emergencyPriority, setEmergencyPriority] = useState(true);

  const activeLanes = useMemo<Lane[]>(() => {
    if (!isRunning) {
      return lanes.map((lane) => ({ ...lane, signal: "red" }));
    }

    if (mode === "manual") {
      return lanes.map((lane) => {
        let signal: SignalState = "red";
        if (lane.id === manualLane) signal = "green";
        if (lane.id === (manualLane + 1) % lanes.length) signal = "amber";
        return { ...lane, signal };
      });
    }

    return lanes;
  }, [isRunning, manualLane, mode]);

  const selectedTab = feedTabs.find((tab) => tab.id === selectedFeed) ?? feedTabs[0];
  const totalVehicles = activeLanes.reduce((sum, lane) => sum + lane.vehicles, 0);
  const bestLane = activeLanes.reduce((best, lane) => {
    return lane.priorityScore > best.priorityScore ? lane : best;
  }, activeLanes[0]);
  const maxScore = Math.max(...activeLanes.map((lane) => lane.priorityScore), 1);
  const learningProgress = Math.min(Math.round((112 / learningFrames) * 100), 100);
  const zoneLabels = manualZones.map((zone) => {
    const totals = zone.points.reduce(
      (acc, [x, y]) => ({ x: acc.x + x, y: acc.y + y }),
      { x: 0, y: 0 }
    );
    return {
      ...zone,
      x: totals.x / zone.points.length,
      y: totals.y / zone.points.length
    };
  });

  return (
    <main className="app-shell">
      <aside className="side-rail" aria-label="Main navigation">
        <div className="brand-mark" aria-hidden="true">
          <Zap size={22} />
        </div>
        <nav className="rail-nav">
          {navItems.map((item, index) => {
            const Icon = item.icon;
            return (
              <button
                className={`rail-button${index === 0 ? " is-active" : ""}`}
                key={item.label}
                type="button"
                aria-label={item.label}
                title={item.label}
              >
                <Icon size={20} />
              </button>
            );
          })}
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">AI intersection control</p>
            <h1>Smart Traffic Command</h1>
          </div>

          <div className="topbar-actions">
            <div className="system-pill">
              <span className="pulse-dot" />
              {isRunning ? "Active" : "Paused"}
            </div>
            <IconButton label="Refresh stream">
              <RefreshCw size={18} />
            </IconButton>
            <IconButton label="Download report">
              <Download size={18} />
            </IconButton>
            <IconButton label="Notifications">
              <Bell size={18} />
            </IconButton>
          </div>
        </header>

        <section className="metric-grid" aria-label="System metrics">
          <MetricCard
            icon={<ShieldCheck size={18} />}
            label="AI State"
            value={mode === "autonomous" ? "Autonomous" : "Manual"}
            detail={`${systemConfig.clusterCount} learned routes`}
            tone="good"
          />
          <MetricCard
            icon={<Activity size={18} />}
            label="Vehicles"
            value={String(totalVehicles)}
            detail="tracked in frame"
          />
          <MetricCard
            icon={<Gauge size={18} />}
            label="Priority"
            value={bestLane.priorityScore.toFixed(1)}
            detail={bestLane.name}
            tone={bestLane.emergency ? "danger" : "warn"}
          />
          <MetricCard
            icon={<Timer size={18} />}
            label="Green Cycle"
            value={`${greenSeconds}s`}
            detail={`${bestLane.avgWait} avg wait`}
          />
        </section>

        <section className="content-grid">
          <section className="panel live-panel" aria-labelledby="live-title">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Camera stream</p>
                <h2 id="live-title">Kavakli junction</h2>
              </div>
              <div className="tab-list" role="tablist" aria-label="Feed view">
                {feedTabs.map((tab) => (
                  <button
                    className={`tab-button${selectedFeed === tab.id ? " is-active" : ""}`}
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={selectedFeed === tab.id}
                    onClick={() => setSelectedFeed(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="video-frame">
              <img src={selectedTab.asset} alt={`${selectedTab.label} traffic camera feed`} />
              <div className="feed-badge">
                <Camera size={16} />
                1280p
              </div>
              <div className="traffic-light-stack" aria-label="Current signal state">
                {(["red", "amber", "green"] as SignalState[]).map((state) => (
                  <span
                    key={state}
                    className={`light-dot ${state}${bestLane.signal === state ? " is-on" : ""}`}
                  />
                ))}
              </div>
              {selectedFeed === "zones" && (
                <svg className="zone-overlay" viewBox="0 0 1280 720" aria-hidden="true">
                  {manualZones.map((zone, index) => (
                    <polygon
                      className={`zone-polygon zone-${index}`}
                      key={zone.name}
                      points={zone.points.map(([x, y]) => `${x},${y}`).join(" ")}
                    />
                  ))}
                </svg>
              )}
            </div>
          </section>

          <aside className="panel controls-panel" aria-labelledby="control-title">
            <div className="panel-header compact">
              <div>
                <p className="eyebrow">Signal control</p>
                <h2 id="control-title">Dispatch</h2>
              </div>
              <IconButton
                label={isRunning ? "Pause controller" : "Start controller"}
                active={isRunning}
                onClick={() => setIsRunning((value) => !value)}
              >
                {isRunning ? <CirclePause size={18} /> : <CirclePlay size={18} />}
              </IconButton>
            </div>

            <div className="segmented-control" aria-label="Controller mode">
              <button
                className={mode === "autonomous" ? "is-active" : ""}
                type="button"
                aria-pressed={mode === "autonomous"}
                onClick={() => setMode("autonomous")}
              >
                Auto
              </button>
              <button
                className={mode === "manual" ? "is-active" : ""}
                type="button"
                aria-pressed={mode === "manual"}
                onClick={() => setMode("manual")}
              >
                Manual
              </button>
            </div>

            <label className="field">
              <span>Manual lane</span>
              <select
                value={manualLane}
                disabled={mode !== "manual"}
                onChange={(event) => setManualLane(Number(event.target.value))}
              >
                {lanes.map((lane) => (
                  <option value={lane.id} key={lane.id}>
                    {lane.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Green cycle</span>
              <input
                type="range"
                min="20"
                max="90"
                value={greenSeconds}
                onChange={(event) => setGreenSeconds(Number(event.target.value))}
              />
            </label>

            <label className="field">
              <span>Learning frames</span>
              <input
                type="number"
                min="80"
                max="300"
                value={learningFrames}
                onChange={(event) => setLearningFrames(Number(event.target.value))}
              />
            </label>

            <label className="toggle-row">
              <span>
                <Siren size={17} />
                Emergency priority
              </span>
              <input
                type="checkbox"
                checked={emergencyPriority}
                onChange={(event) => setEmergencyPriority(event.target.checked)}
              />
            </label>
          </aside>
        </section>

        <section className="lower-grid">
          <section className="panel" aria-labelledby="lanes-title">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Weighted routing</p>
                <h2 id="lanes-title">Lane priority</h2>
              </div>
              <span className="mini-pill">
                <GitBranch size={15} />
                K-Means
              </span>
            </div>
            <div className="lane-list">
              {activeLanes.map((lane) => (
                <article className="lane-row" key={lane.id}>
                  <div className="lane-main">
                    <span className={`signal-dot ${signalTone[lane.signal]}`} />
                    <div>
                      <h3>{lane.name}</h3>
                      <p>{lane.direction}</p>
                    </div>
                  </div>
                  <div className="score-track" aria-label={`${lane.name} score`}>
                    <span style={{ width: `${(lane.priorityScore / maxScore) * 100}%` }} />
                  </div>
                  <div className="lane-meta">
                    <strong>{lane.priorityScore.toFixed(1)}</strong>
                    <span>{lane.vehicles} vehicles</span>
                    <span>{lane.trend}</span>
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="panel" aria-labelledby="learning-title">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Model state</p>
                <h2 id="learning-title">Learning</h2>
              </div>
              <span className="mini-pill">
                <SlidersHorizontal size={15} />
                {systemConfig.model}
              </span>
            </div>

            <div className="learning-ring" style={{ "--progress": `${learningProgress}%` } as CSSProperties}>
              <strong>{learningProgress}%</strong>
              <span>trained</span>
            </div>

            <div className="config-table">
              {vehicleWeights.map((item) => (
                <div key={item.id}>
                  <span>{item.label}</span>
                  <strong>{item.weight.toFixed(1)}x</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="panel zone-panel" aria-labelledby="zones-title">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Manual ROI</p>
                <h2 id="zones-title">Zone map</h2>
              </div>
              <span className="mini-pill">{systemConfig.resolution}</span>
            </div>

            <svg className="zone-map" viewBox="0 0 1280 720" role="img" aria-label="Manual traffic zones">
              <rect width="1280" height="720" rx="22" />
              <path d="M0 456 C260 360 482 316 686 300 C885 284 1032 304 1280 392" />
              <path d="M708 0 C670 170 641 320 590 720" />
              <path d="M956 0 C962 226 1026 418 1168 720" />
              {manualZones.map((zone, index) => (
                <polygon
                  className={`zone-polygon zone-${index}`}
                  key={zone.name}
                  points={zone.points.map(([x, y]) => `${x},${y}`).join(" ")}
                />
              ))}
              {zoneLabels.map((zone) => (
                <text x={zone.x} y={zone.y} key={zone.name}>
                  {zone.label}
                </text>
              ))}
            </svg>
          </section>

          <section className="panel" aria-labelledby="events-title">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Operations</p>
                <h2 id="events-title">Timeline</h2>
              </div>
              <span className={`risk-pill${emergencyPriority ? " armed" : ""}`}>
                <AlertTriangle size={15} />
                {emergencyPriority ? "Armed" : "Normal"}
              </span>
            </div>

            <div className="event-list">
              {activityLog.map((event) => (
                <article className="event-item" key={`${event.time}-${event.title}`}>
                  <time>{event.time}</time>
                  <div>
                    <h3>{event.title}</h3>
                    <p>{event.detail}</p>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </section>
      </section>
    </main>
  );
}
