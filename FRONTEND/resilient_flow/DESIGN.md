# Design System Specification: Editorial Resilience

## 1. Overview & Creative North Star
**Creative North Star: "The Sentinel of Softness"**

In high-stress disaster management environments, traditional UI often leans into "hard" brutalism—heavy borders, sharp corners, and alarming high-contrast grids. This design system rejects that anxiety-inducing aesthetic. Instead, we embrace a **Soft Editorial** approach. 

The system treats data as a curated narrative. By utilizing intentional asymmetry, expansive negative space, and organic "liquid" containers, we transform complex ML and GIS data into a calm, authoritative experience. We break the "template" look by layering surfaces like stacked sheets of architectural vellum, ensuring that even in a crisis, the interface remains a source of clarity rather than a source of chaos.

---

## 2. Colors & Tonal Architecture
The palette is built on a sophisticated foundation of deep charcoals and muted slate, allowing semantic alerts to "glow" rather than "shout."

### The "No-Line" Rule
**Explicit Instruction:** 1px solid borders for sectioning are strictly prohibited. 
Structure is defined through **Tonal Shifting**. To separate a sidebar from a main map view, transition from `surface` (#121416) to `surface-container-low` (#1a1c1e). The eye perceives the change in depth naturally, maintaining a premium, "soft" feel.

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of materials. 
- **Base Layer:** `surface-dim` or `surface`.
- **Secondary Modules:** `surface-container-low`.
- **Primary Interaction Cards:** `surface-container-high` or `highest`.
By nesting a `surface-container-lowest` card inside a `surface-container` section, you create a recessed "well" effect that draws the eye without structural clutter.

### The Glass & Gradient Rule
To elevate the "tech-forward" GIS aesthetic:
- **Floating HUDs:** Use `surface` colors at 70% opacity with a `24px` backdrop-blur. 
- **Signature CTAs:** Apply a linear gradient from `primary` (#98cbff) to `primary-container` (#00629d) at a 135-degree angle. This provides a "lithographic" soul to interactive elements.

---

### 3. Typography: The Editorial Voice
Our typography balances the high-performance utility of **Inter** with the geometric character of **Manrope** and **Plus Jakarta Sans**.

- **Display & Headlines (Manrope):** These are our "Command" fonts. Used for large data points and urgent headings. The wide apertures of Manrope ensure legibility even when blurred or viewed from a distance.
- **Titles & Body (Inter):** The "Workhorse." Inter’s tall x-height provides maximum readability for technical field reports and ML-generated insights.
- **Labels (Plus Jakarta Sans):** Used for micro-data and GIS tags. The modern, tech-forward curves of Jakarta Sans complement the "soft" aesthetic of the containers.

**Scale Philosophy:** Use high-contrast scaling. A `display-lg` (3.5rem) title should sit near a `body-md` (0.875rem) description to create a sophisticated, editorial hierarchy that prioritizes information at a glance.

---

## 4. Elevation & Depth
Depth is not a shadow; it is a state.

- **The Layering Principle:** Avoid elevation shadows for static cards. Use the `surface-container` tiering. 
- **Ambient Shadows:** For "Floating" elements (e.g., a critical alert modal), use an "Atmospheric Shadow." 
    - *Values:* `0px 20px 40px`
    - *Color:* `on-surface` (#e2e2e5) at **4% opacity**. This mimics natural light diffusion in a smoke-filled or low-light environment.
- **The "Ghost Border" Fallback:** If a GIS element requires a boundary against a complex map background, use the `outline-variant` (#40484f) at **15% opacity**. Never use 100% opaque lines.
- **Glassmorphism:** Navigation rails and map overlays must use a `surface-variant` with a 60% opacity and a `16px` blur to allow the GIS data to "ghost" through the UI.

---

## 5. Components & Primitives

### Buttons: The "Pill" Interaction
- **Primary:** Large border-radius (`full` or `xl: 3rem`). Use the Primary-to-Container gradient.
- **Secondary:** `surface-container-highest` background. No border.
- **Tertiary:** Text-only with `label-md` styling.
- *Interaction:* On hover, buttons should subtly expand (1.02x scale) with a `300ms` ease-out transition.

### Input Fields: Soft Enclosures
- Use `surface-container-low` for the field body. 
- Border-radius must be `md: 1.5rem`.
- **Active State:** Instead of a thick border, use a 2px "inner glow" using the `surface-tint`.

### Cards & Lists: The Negative Space Rule
- **Forbid Dividers:** Do not use lines to separate list items. Use a `1.5rem` vertical spacing (padding) and a subtle background shift on hover.
- **Soft Corners:** Cards must use `lg: 2rem` or `xl: 3rem` border-radius to reinforce the "soft" aesthetic.

### Disaster Semantic Chips
- **Critical (Error):** Use `error_container` (#93000a) with `on_error_container` text.
- **Warning (Tertiary):** Use `tertiary_container` (#ba1723).
- **Safe (Primary):** Use `primary_container`.
- *Visual Style:* Chips are "Full" radius (9999px) and should have a subtle pulse animation for "Critical" levels.

---

## 6. Do’s and Don’ts

### Do:
- **Do** use asymmetric layouts. Place a large GIS map on the left with a floating, softly rounded data panel overlapping it by 40px.
- **Do** lean into white space. Give ML insights room to breathe.
- **Do** use `surface-container-lowest` for the darkest parts of the Dark Mode UI to create a "black hole" effect for focal data.

### Don't:
- **Don't** use sharp 90-degree corners. Even "small" components must have at least an `sm: 0.5rem` radius.
- **Don't** use pure black (#000) or pure white (#FFF). Use the provided `surface` and `on-surface` tokens to maintain the "soft" tonal range.
- **Don't** use standard "drop shadows." If it doesn't look like frosted glass or layered paper, it doesn't belong in this system.
- **Don't** use lines to separate content. If the content feels cluttered, increase the spacing scale rather than adding a divider.

---

## 7. Interactive Feel
Every interaction must feel "intentional." 
- **Transitions:** Use `cubic-bezier(0.34, 1.56, 0.64, 1)` for all modal entrances. This creates a soft "bounce" that feels tactile and responsive.
- **Staggered Reveals:** When a dashboard loads, data cards should fade in and slide up (20px) with a 50ms stagger between each element. This communicates the "ML processing" aspect of the platform.