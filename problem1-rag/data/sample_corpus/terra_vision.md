# AI Integration Guide - TerraVision

## Overview

TerraVision is an urban sustainability planning platform that leverages **Google Gemini AI** to provide intelligent green corridor suggestions, street transformation analysis, and urban environmental metrics. This document provides complete technical details about the AI implementation.

---

## 1. AI Model Used: Google Gemini

### Primary Model Details
- **Model Name**: `gemini-2.5-flash` (default for text/analysis)
- **Model Variant**: Google Generative AI SDK v1.39.0
- **API Provider**: Google AI Studio
- **Language**: TypeScript/JavaScript via `@google/genai` SDK

### Image Generation Models (Fallback Chain)
The application attempts multiple image generation models in order:
1. `gemini-2.5-flash-image` (Primary)
2. `gemini-3.1-flash-image-preview` (Secondary)
3. `gemini-3-pro-image-preview` (Tertiary)

**Why Multiple Models?**
- Different API keys have different quota access
- Provides fallback if primary model is not available
- Ensures robust image generation pipeline

### Model Capabilities Used

| Capability | Purpose | Implementation |
|------------|---------|----|
| **Text Analysis** | Green corridor planning, urban metrics analysis | `analyzeRegionForGreenCorridor()` |
| **Image Generation** | Transform street photos with sustainability interventions | `generateImageWithDetails()` |
| **Vision API** | Analyze uploaded photos for sustainability insights | `analyzeImage()` |
| **JSON Output** | Structured corridor suggestions | Response MIME type: `application/json` |

---

## 2. API Key Configuration

### Setup Steps

1. **Get Your API Key**
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Enable Gemini API access

2. **Store API Key Securely**
   
   Create `.env.local` file in project root:
   ```bash
   VITE_GEMINI_API_KEY=your_api_key_here
   ```

3. **API Key Validation**
   
   The service automatically checks for the key:
   ```typescript
   const apiKey = import.meta.env.VITE_GEMINI_API_KEY || import.meta.env.GEMINI_API_KEY;
   ```

### Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Gemini API key missing" | `VITE_GEMINI_API_KEY` not set | Add to `.env.local` |
| "RESOURCE_EXHAUSTED" | API quota exceeded | Enable paid billing in AI Studio |
| "NOT_FOUND" | Model not available for key | Verify API access in AI Studio |
| "Image generation quota is exhausted" | Image generation not enabled | Enable image models in billing settings |

---

## 3. Core AI Features & Functions

### 3.1 Region Analysis & Green Corridor Suggestion
**Function**: `analyzeRegionForGreenCorridor(centerLat, centerLng, activeLayers, regionData)`

**Purpose**: Analyzes a clicked region on the map and suggests an optimal green corridor placement.

**Input Parameters**:
- `centerLat` / `centerLng`: Clicked point coordinates (decimal degrees, e.g., 28.6139)
- `activeLayers`: Array of active map layers (e.g., `["trees", "solar", "ev_stations"]`)
- `regionData`: Region metrics including tree count, EV stations, solar panels

**AI Prompt Template**:
```
Given urban data, suggest a green corridor plan in strict JSON format.
Center: {{centerLat}}, {{centerLng}}
Active layers: {{activeLayers}}
Region data summary: {{regionData}}

Constraints:
- Keep every path coordinate strictly within bounds: lat [minLat, maxLat], lng [minLng, maxLng]
- Avoid dense tree-cover patches and route beside them, not through them
- Avoid built-up footprints/rooftop zones; do not pass through likely buildings
- Prefer simple street-aligned paths with only 3 to 5 total points and clear bends
```

**AI Output (JSON)**:
```json
{
  "corridorPath": [[28.614, 77.208], [28.615, 77.209], ...],
  "corridorType": "East-West Green Connector",
  "reasoning": "Connects park systems while avoiding built structures...",
  "features": ["Shade trees", "Bike lanes", "Permeable pathways", "Solar lighting"]
}
```

**Post-Processing Pipeline**:
1. **Path Simplification** → Max 5 waypoints via `simplifyToMaxPoints()`
2. **Boundary Clamping** → All coordinates constrained to region bounds via `clampToRegion()`
3. **Obstacle Avoidance** → Pathfinding nudges around trees and solar features
4. **Deduplication** → Removes points closer than 0.00008 decimal degrees (~9m)

**Return Object**:
```typescript
{
  success: boolean;
  corridorPath: [lat, lng][]; // 5-point max simplified path
  corridorType: string;       // e.g., "North-South Green Spine"
  reasoning: string;          // Why this path is optimal
  features: string[];         // Implementation features
  error?: string;            // Error message if failed
}
```

---

### 3.2 Region Metrics Analysis
**Function**: `analyzeRegionData(centerLat, centerLng)`

**Purpose**: Analyzes urban infrastructure metrics for a clicked region without AI.

**Process**:
1. Loads three GeoJSON datasets (trees, EV stations, solar panels)
2. Counts features within region boundary
3. Calculates green cover percentage
4. Computes heat/green scores (0-100 scale)

**Output Metrics**:
```typescript
{
  name: string;                    // Region name (e.g., "Delhi Cantonment")
  heatScore: number;              // 0-100, higher = more heat stress
  greenScore: number;             // 0-100, higher = more vegetation
  stats: {
    trees: number;                 // Tree equivalent count
    ev: number;                    // EV charging stations
    solar: number;                 // Solar panel installations
    area: string;                  // Region area in km²
  }
}
```

**Score Calculation Formula**:
```
greenness = greenCoverPct × 0.7 + solar × 2.5 + ev
pressure = max(0, 40 - (greenCoverPct × 0.5 + solar))

greenScore = clamp(12 + greenness × 1.2, 12, 98)
heatScore = clamp(90 - greenCoverPct × 0.75 - solar × 1.1 + pressure × 0.25, 24, 96)
```

---

### 3.3 City-Wide Corridor Suggestions
**Function**: `suggestCityGreenCorridors()`

**Purpose**: Generates city-wide green corridor network recommendations.

**Implementation**:
- Uses grid-based layout (5×5 cells across city bounds)
- Generates 6 predefined corridor types
- Returns prioritized corridor network

**Generated Corridors** (6 total):
1. **East-West Green Spine (High Priority)** - Major connectivity
2. **East-West Secondary Link (Medium Priority)** - Supporting flow
3. **North-South Green Spine (High Priority)** - Ecological continuity
4. **North-South Secondary (Medium Priority)** - Neighborhood connectivity
5. **Connector Corridor 1 (Low Priority)** - L-shaped local connector
6. **Connector Corridor 2 (Low Priority)** - Network completion

**Return Object**:
```typescript
{
  success: boolean;
  corridors: {
    id: string;
    path: [lat, lng][];
    corridorType: string;
    priority: "High" | "Medium" | "Low";
    reasoning: string;
    zones: string[];
  }[];
  citySummary: string;
  error?: string;
}
```

---

### 3.4 Street Image Transformation with AI
**Function**: `generateImageWithDetails(file, prompt)`

**Purpose**: Uses Gemini to transform street photos with sustainability interventions (trees, shade structures, green solutions).

**Input**:
- `file`: Uploaded JPEG/PNG image
- `prompt`: Transformation instructions (e.g., "Add 50 shade trees, green corridors, solar panels")

**Image Processing Pipeline**:
1. Convert image to Base64
2. Send to Gemini with transformation prompt
3. Request `responseModalities: ["TEXT", "IMAGE"]`
4. Extract generated image and text analysis
5. Fallback to multiple models if needed

**Output**:
```typescript
{
  success: boolean;
  prompt: string;
  analysis: string;              // AI's text analysis of changes
  generated_image_url: string;   // Data URL of transformed image
  original_image: string;        // Original image URL (fallback)
  error?: string;
}
```

**Example Prompt Used**:
```
Transform this uploaded urban street photo using the instructions below while 
preserving road perspective and major structures:
{{user_prompt}}

Also include a concise analysis of key sustainability improvements.
```

---

### 3.5 Image Analysis
**Function**: `analyzeImage(file, prompt)`

**Purpose**: Analyzes uploaded photos using Gemini Vision API.

**Input**:
- `file`: Uploaded image
- `prompt`: Analysis question (e.g., "What green infrastructure exists here?")

**Output**:
```typescript
{
  success: boolean;
  prompt: string;
  analysis: string;  // AI's detailed analysis
  error?: string;
}
```

---

## 4. Geospatial Data Integration

### GeoJSON Datasets

The AI processes three GeoJSON datasets:

| Dataset | Path | Purpose | Geometry Type |
|---------|------|---------|---|
| **Tree Cover** | `/public/data/tree_cover_vegetation.geojson` | Green space identification | Polygon/Point |
| **Solar Panels** | `/public/data/solarPanelsDATA.geojson` | Built infrastructure proxy | Point/Polygon |
| **EV Stations** | `/public/data/ev_charging_stations.geojson` | Charging infrastructure | Point |

### Data Processing Functions

#### `getTreePointsInRegion(centerLat, centerLng)`
Extracts tree centroids within a 0.004° radius region (~444m at equator)

```typescript
function getTreePointsInRegion(centerLat, centerLng): Promise<[number, number][]>
// Returns: Array of [latitude, longitude] for each tree
```

#### `getSolarPointsInRegion(centerLat, centerLng)`
Extracts solar panel centroids as proxy for built areas

```typescript
function getSolarPointsInRegion(centerLat, centerLng): Promise<[number, number][]>
// Returns: Array of [latitude, longitude] for each solar feature
```

### Region Boundary Constants

```typescript
const REGION_HALF_DELTA = 0.004;  // 0.008° × 0.008° region
const REGION_AREA_KM2 = (2 * REGION_HALF_DELTA * 111.32) * (2 * REGION_HALF_DELTA * 111.32);
// ≈ 0.79 km²
```

---

## 5. Green Corridor Path Processing

### Advanced Geospatial Functions

#### `clampToRegion(centerLat, centerLng, point)`
Ensures all corridor points stay within region bounds.

```typescript
// Input: point = [28.6200, 77.2100]
// Output: point clamped to [minLat, maxLat] × [minLng, maxLng]
```

#### `buildSimpleRoadLikePath(centerLat, centerLng, inputPath)`
Generates L-shaped or orthogonal bend structure (max 5 points).

**Algorithm**:
1. Extract start, end, and guide (middle) points from input
2. Determine primary bend direction (horizontal vs. vertical)
3. Create 5-point path: start → bend1 → guide → bend2 → end
4. All points clamped to region bounds

**Example Output**:
```
Path: [[28.614, 77.208], [28.614, 77.209], [28.6142, 77.2092], [28.615, 77.209], [28.615, 77.210]]
     start              bend1              guide              bend2            end
```

#### `pushPointAwayFromTrees(centerLat, centerLng, point, obstaclePoints)`
Nudges waypoints away from obstacles (trees and solar features).

**Algorithm**:
1. Find nearest obstacle to point
2. If distance < SAFE_DISTANCE (0.00030 decimal degrees):
   - Calculate direction vector away from obstacle
   - Move point proportionally based on proximity
   - Re-clamp to region bounds
3. Repeat up to 3 iterations until safe

**Safe Distance**: 0.00030° ≈ 33 meters at equator

#### `simplifyToMaxPoints(path, maxPoints = 5)`
Reduces path complexity via linear interpolation sampling.

**Algorithm**:
1. If path already has ≤ maxPoints: return as-is
2. Otherwise, evenly sample maxPoints from path
3. Deduplicate closely-spaced points
4. Return simplified path

---

## 6. Cost Model Integration

### Cost Per Unit (INR - Indian Rupees)

| Intervention Type | Cost Per Unit | Notes |
|------------------|--------------|-------|
| **Shade Trees** | ₹4,500/tree | Mature specimens, maintenance |
| **Green Belt** | ₹500/sq.m | Ground cover, shrubs |
| **Vegetative Canopy** | ₹400/sq.m | Dense foliage layers |
| **Cycle Lanes** | ₹150/meter | Infrastructure markup |
| **Shade Structures** | ₹800/unit | Pergolas, pavilions |
| **Community Gardens** | ₹600/plot | Medium-sized plots |

### Dynamic Cost Calculation

The Street Tool calculates costs in real-time:

```typescript
plantCounts = {
  trees: 50,
  greenBelt: 100,
  vegetation: 75,
  cycleLanes: 200,
  shade: 10,
  gardens: 5
}

totalCost = (50 × 4500) +      // ₹225,000
            (100 × 500) +      // ₹50,000
            (75 × 400) +       // ₹30,000
            (200 × 150) +      // ₹30,000
            (10 × 800) +       // ₹8,000
            (5 × 600)          // ₹3,000
          = ₹346,000
```

---

## 7. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     TerraVision Frontend                    │
│  (React + Leaflet Map)                                      │
└────────────┬──────────────────────────────────────────────────┘
             │
             ├─────────────────────────┐
             ▼                         ▼
    ┌────────────────┐        ┌──────────────────┐
    │  MapExplorer   │        │  StreetTool      │
    │  (Analysis UI) │        │  (Metrics UI)    │
    └────────┬───────┘        └──────┬───────────┘
             │                       │
             └──────────┬────────────┘
                        ▼
          ┌─────────────────────────────┐
          │   geminiService.ts           │
          │  (AI & Geo Processing)       │
          └─────────┬───────────┬────────┘
                    │           │
        ┌───────────┘           └──────────────┐
        │                                      │
        ▼                                      ▼
    ┌─────────────────┐              ┌──────────────────┐
    │ Gemini AI API   │              │ GeoJSON Datasets │
    │ (Google Cloud)  │              │ (/public/data)   │
    │                 │              │                  │
    │ • gemini-2.5    │              │ • Trees (GeoJSON)│
    │ • Image Gen     │              │ • Solar (GeoJSON)│
    │ • Vision        │              │ • EV Stations    │
    │ • JSON Response │              │                  │
    └─────────────────┘              └──────────────────┘
```

---

## 8. Request/Response Flow Example

### User Action: Click on Map Region

```
1. User clicks on map → MapExplorer component
2. Event handler calls:
   - analyzeRegionData(lat, lng)
   - analyzeRegionForGreenCorridor(lat, lng, layers, regionData)

3. Service Processing:
   ├─ Load GeoJSON datasets (cached)
   ├─ Count features in region (trees, EV, solar)
   ├─ Calculate scores and metrics
   └─ Send structured prompt to Gemini AI

4. Gemini AI Response:
   └─ Returns JSON with corridorPath, type, reasoning, features

5. Path Post-Processing:
   ├─ Clamp to region bounds
   ├─ Simplify to 5 points max
   ├─ Nudge away from trees/solar
   └─ Deduplicate closely-spaced points

6. Render on Map:
   └─ Draw corridor polyline, add interactive drawer
```

---

## 9. Technical Specifications

### Coordinate System
- **Format**: Decimal degrees (WGS84)
- **Example**: Delhi center [28.6139, 77.2090]
- **Precision**: 5 decimal places (~1.1 meters)

### Distance Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| **Dedup Distance** | 0.00008° | ~9 meters - remove micro-oscillations |
| **Safe Distance** | 0.00030° | ~33 meters - obstacle avoidance buffer |
| **Detour Offset** | 0.0003° | ~33 meters - nudge magnitude |
| **Region Radius** | 0.004° | ~444 meters - analysis boundary |

### Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Load GeoJSON (cached) | <100ms | First call, then cached |
| Analyze Region Metrics | ~200ms | Local computation |
| AI Corridor Suggestion | 2-5s | Network latency + model inference |
| Image Generation | 10-30s | Image model + processing |
| Image Analysis | 5-10s | Vision model processing |

---

## 10. Error Handling & Fallbacks

### AI Failure Scenarios

#### Scenario 1: API Key Missing
```
Error: "Gemini API key missing. Set VITE_GEMINI_API_KEY..."
Fallback: Show error message to user, disable AI features
```

#### Scenario 2: API Quota Exceeded
```
Error: "RESOURCE_EXHAUSTED" or "Quota exceeded"
Fallback: Show error message, suggest upgrading to paid plan
```

#### Scenario 3: Invalid JSON Response
```
Error: JSON.parse() fails on AI response
Fallback: Use hardcoded diagonal corridor path with clamping
```

#### Scenario 4: Image Generation Unavailable
```
Error: Image generation model not available
Fallback: Try next model in chain (3 total models attempted)
```

### Fallback Corridor Path
When AI fails, a pre-computed diagonal path is used:
```typescript
const fallbackPath: [number, number][] = [
  [centerLat + 0.003, centerLng - 0.0025],
  [centerLat + 0.001, centerLng - 0.0005],
  [centerLat - 0.001, centerLng + 0.001],
  [centerLat - 0.003, centerLng + 0.0025],
];
```

---

## 11. Environment Variables

### Required
```bash
VITE_GEMINI_API_KEY=<your-gemini-api-key>
```

### Optional Variants
```bash
GEMINI_API_KEY=<fallback-key>  # Fallback if VITE_ version missing
```

### Configure in `.env.local`
```bash
# Create .env.local in project root
VITE_GEMINI_API_KEY=AIza...your_actual_key_here...
```

---

## 12. Usage Examples

### Example 1: Analyze a Region

```typescript
import { analyzeRegionData, analyzeRegionForGreenCorridor } from '@/services/geminiService';

// User clicks on New Delhi location [28.6139, 77.2090]
const metrics = await analyzeRegionData(28.6139, 77.2090);
console.log(metrics);
// Output:
// {
//   name: "Delhi Cantonment",
//   heatScore: 72,
//   greenScore: 48,
//   stats: { trees: 245, ev: 12, solar: 8, area: "0.79 km2" }
// }

// Get corridor suggestion
const corridor = await analyzeRegionForGreenCorridor(
  28.6139, 77.2090,
  ["trees", "solar", "ev_stations"],
  metrics
);
console.log(corridor);
// Output:
// {
//   success: true,
//   corridorPath: [[28.614, 77.208], [28.614, 77.209], ...],
//   corridorType: "East-West Green Spine",
//   reasoning: "Connects park systems...",
//   features: ["Shade trees", "Bike lanes", ...]
// }
```

### Example 2: Transform a Street Photo

```typescript
import { generateImageWithDetails } from '@/services/geminiService';

const file = document.getElementById('photoInput').files[0]; // User uploads photo
const prompt = "Add 30 shade trees, green corridor with cycle lane, community garden, solar lighting";

const result = await generateImageWithDetails(file, prompt);
if (result.success) {
  console.log("AI Analysis:", result.analysis);
  console.log("Transformed image:", result.generated_image_url);
  // Display before/after slider in UI
} else {
  console.error("Error:", result.error);
  // Show original image as fallback
}
```

### Example 3: Get City-Wide Corridors

```typescript
import { suggestCityGreenCorridors } from '@/services/geminiService';

const cityPlan = await suggestCityGreenCorridors();
console.log(`Generated ${cityPlan.corridors.length} city corridors`);

cityPlan.corridors.forEach(corridor => {
  console.log(`${corridor.corridorType} (${corridor.priority}):`, corridor.path);
});
```

---

## 13. Best Practices

### 1. API Key Security
- ✅ Store in `.env.local` (gitignored)
- ✅ Never commit API keys to version control
- ✅ Use different keys for dev/prod if possible
- ❌ Don't hardcode keys in source

### 2. Quota Management
- Monitor API usage in Google AI Studio
- Set appropriate quota limits
- Consider caching GeoJSON datasets locally
- Cache AI responses if possible

### 3. User Experience
- Show loading spinners during AI requests (2-5s wait)
- Provide fallback paths when AI unavailable
- Explain why requests are slow
- Handle errors gracefully with helpful messages

### 4. Performance Optimization
- GeoJSON datasets are cached after first load
- Region analysis is pre-computed (no AI calls)
- Corridor paths simplified to max 5 points
- Use deduplication to avoid rendering micro-oscillations

---

## 14. Troubleshooting

### Issue: "API key missing" error appears

**Solution**:
1. Create `.env.local` in project root
2. Add: `VITE_GEMINI_API_KEY=your_key_here`
3. Restart dev server: `npm run dev`

### Issue: Image generation not working

**Solution**:
1. Check API Studio has "Image Generation" enabled
2. Check if you're on a paid billing plan
3. Try uploading a different image format (JPEG/PNG only)
4. Check browser console for detailed error

### Issue: Corridor suggestions appear random/incorrect

**Solution**:
1. Check active layers (trees, solar, EV should be enabled)
2. Ensure region has adequate data (some areas may be sparse)
3. Verify coordinates are valid (0-90 for lat, -180-180 for lng)
4. Check that GeoJSON files loaded successfully (network tab)

### Issue: Very slow responses (>10s)

**Solution**:
1. Check network latency to Google APIs
2. Verify API quota not exceeded
3. Try simpler prompts (fewer constraints)
4. Consider enabling paid tier for faster inference

---

## 15. Dependencies

```json
{
  "@google/genai": "^1.39.0",
  "firebase": "^12.8.0",
  "react": "^19.2.4",
  "leaflet": "1.9.4"
}
```

### Key Packages
- **@google/genai**: Official Google Generative AI SDK
- **firebase**: Authentication & database services
- **leaflet**: Map rendering and geospatial visualization

---

## 16. Deployment Considerations

### Frontend Deployment
1. API key must be in deployment environment variables
2. Vite build optimizes bundle size
3. GeoJSON files served from `/public/data/`

### Environment Setup
```bash
# Production deployment requires:
VITE_GEMINI_API_KEY=production_key_here
VITE_FIREBASE_PROJECT_ID=terravision-b0eb4
```

### Rate Limiting
- Each user can have multiple parallel requests
- Consider implementing request debouncing in UI
- Cache results to minimize redundant API calls

---

## 17. Future Enhancements

### Planned Features
- [ ] Building footprint polygon data (current: solar proxy only)
- [ ] Historical trend analysis (AI identifies improvements over time)
- [ ] Multi-language corridor descriptions
- [ ] Real-time AQI data integration with corridor suggestions
- [ ] Advanced constraint optimization (budget, timeline, maintenance)

### Potential Improvements
- Integrate with OpenStreetMap building data
- Add real-time cost estimation based on vendor quotes
- Implement collaborative corridor planning (multi-user)
- Add climate zone-specific plant recommendations

---

## Summary

**TerraVision leverages Google's Gemini AI (gemini-2.5-flash) to intelligently analyze urban regions and suggest optimal green corridor placements.** The system combines:

- **AI Capabilities**: Text analysis, image generation, vision API
- **Geospatial Processing**: GeoJSON parsing, boundary checking, obstacle avoidance
- **Cost Modeling**: Real-time ROI calculations in Indian Rupees
- **Resilience**: Multiple fallbacks, graceful error handling, cached data

This comprehensive integration enables urban planners to make data-driven decisions about green infrastructure investments with a simple click-to-analyze interface.

---

**Last Updated**: March 2026  
**Version**: 1.0  
**Project**: TerraVision - Sustainable City Planner
