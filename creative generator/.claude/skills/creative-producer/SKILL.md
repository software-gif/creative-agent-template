# Creative Producer

> Generiert Static Ads via Gemini 3.1 Flash Image Generation basierend auf Ad Angles, Brand Guidelines und Produktbildern.

## Problem
Manuell Static Ads zu erstellen ist zeitaufwändig und erfordert Design-Expertise. Der Creative Producer automatisiert die Generierung von brand-konformen Static Ads basierend auf der Andromeda-Diversification-Logik.

## Trigger
Nachdem Angle Generator gelaufen ist und der User die Generierung freigibt.

## Workflow
1. User gibt Anweisung (z.B. "Generiere 30 Ads" oder "Generiere nur für Problem/Pain")
2. Claude wählt passende Angles/Sub-Angles und Produktbilder
3. Claude baut pro Ad einen detaillierten JSON-Prompt
4. Script sendet JSON-Prompt + Produktbild an Gemini 3.1 Flash
5. Generierte Ads werden in `creatives/` gespeichert mit Metadaten
6. Claude gibt Überblick: Angle, Sub-Angle, Format, Vorschau

## Inputs
Liest automatisch:
- `angles/angles.json` — Angles + Sub-Angles + Hooks
- `brand_guidelines.json` — Farben, Fonts, Stilrichtung
- `brand.json` — Brand-Kontext
- `products/products.json` — Produktdaten
- `products/images/<handle>/0.jpg` — Freisteller (Index 0 = sauberes Produktbild)
- `winners/assets/` — Winner Ads als Stil-Referenz

## Outputs
- `creatives/<timestamp>/` — Generierte Ads (4K)
- `creatives/<timestamp>/manifest.json` — Metadaten pro Ad (Angle, Sub-Angle, Format, Prompt)

## Formate
- **4:5** (1536×1920) — Feed
- **9:16** (1080×1920 → 4K: 2160×3840) — Story

## Scripts
- `scripts/main.py` — Orchestrierung: Prompt-Bau, Gemini API Call, Speicherung
- `scripts/prompt_schema.json` — JSON-Prompt-Schema (Single Source of Truth)

## Ausführung
Wird durch Claude gesteuert, nicht direkt ausgeführt. Claude:
1. Wählt Angles/Sub-Angles nach User-Anweisung
2. Generiert JSON-Prompts
3. Ruft `scripts/main.py` mit den Prompts auf

## Dependencies
- Python 3
- `requests` (pip)
- `python-dotenv` (pip)
- `Pillow` (pip) — für Bildverarbeitung
- Gemini API Key in `.env` (`GEMINI_API_KEY`)
- Modell: `gemini-3.1-flash-image-preview`

## Wichtige Regeln für Bild-Prompts

### Produkt NICHT in negativen Szenen
Bei Szenen die etwas Negatives kommunizieren (Problem, Materialermüdung, Pain Points, Unique Mechanism gegen Konkurrenz) darf das beworbene Produkt NIEMALS im Bild erscheinen. Grund: Der Zuschauer assoziiert das Gezeigte mit der Botschaft — das Negative überträgt sich aufs Produkt.
- **Betrifft: Hook, Lead, Unique Mechanism (Problem) — ALLE Szenen vor der Solution/Reveal**
- **Problem-/Negativ-Szenen:** Generische/No-Name-Produkte, abstrakte Grafiken oder typische Konkurrenz-Darstellungen verwenden. Z.B. "abgelatschte generische Sneaker", "no-brand billig-Schuh" — NICHT das beworbene Produkt.
- **Eigenes Produkt erst ab Solution/Product Reveal zeigen** — vorher ist es tabu, auch wenn die Szene "nur" das Problem beschreibt

### Nur echte Produktvarianten
Zeige nur Produktfarben und -varianten die laut `products/products.json` tatsächlich existieren. Erfinde KEINE Farben oder Varianten. Wenn keine Variantendaten vorhanden sind, nur die Hauptfarbe verwenden.

### Hintergründe: Authentisch aber sauber
Der Stil darf "casual" und "unpoliert" sein, aber Hintergründe müssen sauber und aufgeräumt sein. Keine schmutzigen Oberflächen, Krümel, Geschirr-Chaos etc. In Bild-Prompts: "clean background", "tidy surface", "minimal clutter" ergänzen.

### Visuelle Kontinuität über alle Szenen
Alle Bilder eines Storyboards müssen aussehen wie aus EINEM durchgehenden Dreh. Bevor Bild-Prompts erstellt werden:
1. **Setting definieren:** Ort (z.B. "heller Holztisch"), Licht (z.B. "warmes Tageslicht von links"), Farbwelt, Kamera-Stil
2. **Setting-Beschreibung in JEDEN Bild-Prompt einbauen** als gemeinsamen Kontext
3. Perspektiven dürfen variieren (Nah → Halbtotale), aber Ort, Licht und Farbwelt bleiben gleich
4. Besonders bei UGC-Stil: Ein Creator-Typ, ein Ort, eine Kamera-Ästhetik durchgehend

## Verbindungen
- Liest Output von `angle-generator`, `product-scraper`, `ad-library-scraper`
- Referenziert `meta-andromeda` Knowledge
- Referenziert `brand_guidelines.json`
