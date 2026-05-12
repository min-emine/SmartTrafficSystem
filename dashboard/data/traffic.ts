export type SignalState = "green" | "amber" | "red";

export type Lane = {
  id: number;
  name: string;
  direction: string;
  signal: SignalState;
  vehicles: number;
  priorityScore: number;
  occupancy: number;
  avgWait: string;
  trend: string;
  emergency: boolean;
};

export type ManualZone = {
  name: string;
  label: string;
  points: [number, number][];
};

export const systemConfig = {
  streamUrl: "https://content.tvkur.com/l/c77i6m384cnrb6mlji4g/master.m3u8",
  modelPath: "yolo11n.pt",
  resolution: "1280 x 720",
  learningFrames: 150,
  clusterCount: 4,
  model: "YOLOv11 Nano",
  tracker: "ByteTrack"
};

export const vehicleWeights = [
  { id: 2, label: "Car", weight: 1.0 },
  { id: 3, label: "Motorcycle", weight: 1.5 },
  { id: 5, label: "Bus", weight: 4.0 },
  { id: 7, label: "Truck", weight: 1.5 }
];

export const lanes: Lane[] = [
  {
    id: 0,
    name: "West Approach",
    direction: "Bati",
    signal: "green",
    vehicles: 18,
    priorityScore: 32.5,
    occupancy: 78,
    avgWait: "00:42",
    trend: "+14%",
    emergency: false
  },
  {
    id: 1,
    name: "Service Road",
    direction: "Yanyol",
    signal: "red",
    vehicles: 11,
    priorityScore: 18,
    occupancy: 49,
    avgWait: "01:08",
    trend: "-6%",
    emergency: false
  },
  {
    id: 2,
    name: "North Right",
    direction: "Kuzey sag",
    signal: "amber",
    vehicles: 9,
    priorityScore: 21.5,
    occupancy: 57,
    avgWait: "00:55",
    trend: "+4%",
    emergency: true
  },
  {
    id: 3,
    name: "North Left",
    direction: "Kuzey sol",
    signal: "red",
    vehicles: 14,
    priorityScore: 24,
    occupancy: 64,
    avgWait: "01:21",
    trend: "+9%",
    emergency: false
  }
];

export const manualZones: ManualZone[] = [
  {
    name: "bati",
    label: "West",
    points: [
      [86, 253],
      [11, 289],
      [139, 426],
      [350, 347],
      [380, 331],
      [83, 253]
    ]
  },
  {
    name: "yanyol",
    label: "Service",
    points: [
      [277, 253],
      [280, 294],
      [656, 271],
      [711, 245],
      [679, 227],
      [625, 241],
      [279, 252]
    ]
  },
  {
    name: "kuzey sag",
    label: "North R",
    points: [
      [894, 381],
      [589, 336],
      [790, 207],
      [847, 178],
      [919, 186],
      [892, 381]
    ]
  },
  {
    name: "kuzey sol",
    label: "North L",
    points: [
      [960, 184],
      [998, 391],
      [1005, 434],
      [1008, 471],
      [1274, 464],
      [1073, 196],
      [969, 184],
      [958, 185]
    ]
  }
];

export const activityLog = [
  {
    time: "20:31",
    title: "Lane 0 received green",
    detail: "Highest weighted score after bus detection."
  },
  {
    time: "20:29",
    title: "Emergency priority armed",
    detail: "North-right route marked for fast clearance."
  },
  {
    time: "20:27",
    title: "K-Means clusters stable",
    detail: "Four learned routes match configured lane count."
  }
];
