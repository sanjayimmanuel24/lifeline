// Turn the zone GeoJSON into SVG path strings (equirectangular, latitude-corrected) for small maps.
export type Proj = { paths: string[]; width: number; height: number; xy: (lat: number, lon: number) => [number, number] };

export function project(geojson: any, zones: string[], width = 200): Proj {
  const feats = zones.map((z) => geojson.features.find((f: any) => f.properties.zone === z));
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const k = Math.cos((40.65 * Math.PI) / 180);
  const each = (cb: (x: number, y: number) => void) => feats.forEach((f) => polys(f).forEach((ring) => ring.forEach(([lon, lat]) => cb(lon * k, lat))));
  each((x, y) => { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); });
  const s = width / (maxX - minX);
  const height = (maxY - minY) * s;
  const xy = (lat: number, lon: number): [number, number] => [(lon * k - minX) * s, (maxY - lat) * s];
  const paths = feats.map((f) => polys(f).map((ring) => "M" + ring.map(([lon, lat]) => xy(lat, lon).map((v) => v.toFixed(1)).join(",")).join("L") + "Z").join(""));
  return { paths, width, height, xy };
}

function polys(f: any): number[][][] {
  if (!f) return [];
  const g = f.geometry;
  return g.type === "Polygon" ? g.coordinates : g.coordinates.flat();
}

// sequential colour ramp (dark-to-bright) and a diverging one, as CSS colours
export function seq(t: number): string {
  t = Math.max(0, Math.min(1, t));
  const stops = [[24, 32, 43], [40, 90, 160], [62, 166, 255], [255, 200, 90], [255, 77, 94]];
  const p = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), f = p - i;
  const c = stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * f));
  return `rgb(${c.join(",")})`;
}

export function good(t: number): string {   // 0 = bad (red) ... 1 = good (green)
  t = Math.max(0, Math.min(1, t));
  const a = [255, 77, 94], b = [245, 165, 36], c = [47, 191, 113];
  const [x, y, f] = t < 0.5 ? [a, b, t * 2] : [b, c, (t - 0.5) * 2];
  return `rgb(${x.map((v, j) => Math.round(v + (y[j] - v) * f)).join(",")})`;
}
