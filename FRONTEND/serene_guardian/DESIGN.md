---
name: Serene Guardian
colors:
  surface: '#f8f9fe'
  surface-dim: '#d8dade'
  surface-bright: '#f8f9fe'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f4f8'
  surface-container: '#eceef2'
  surface-container-high: '#e6e8ed'
  surface-container-highest: '#e0e2e7'
  on-surface: '#191c1f'
  on-surface-variant: '#40484f'
  inverse-surface: '#2d3134'
  inverse-on-surface: '#eff1f5'
  outline: '#707880'
  outline-variant: '#c0c7d0'
  surface-tint: '#006494'
  primary: '#004b71'
  on-primary: '#ffffff'
  primary-container: '#006494'
  on-primary-container: '#b6ddff'
  inverse-primary: '#8ecdff'
  secondary: '#5a5f62'
  on-secondary: '#ffffff'
  secondary-container: '#dce0e4'
  on-secondary-container: '#5e6367'
  tertiary: '#6a3b00'
  on-tertiary: '#ffffff'
  tertiary-container: '#8b5001'
  on-tertiary-container: '#ffcfa5'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#cbe6ff'
  primary-fixed-dim: '#8ecdff'
  on-primary-fixed: '#001e30'
  on-primary-fixed-variant: '#004b71'
  secondary-fixed: '#dfe3e7'
  secondary-fixed-dim: '#c3c7cb'
  on-secondary-fixed: '#171c1f'
  on-secondary-fixed-variant: '#43474b'
  tertiary-fixed: '#ffdcbf'
  tertiary-fixed-dim: '#ffb872'
  on-tertiary-fixed: '#2d1600'
  on-tertiary-fixed-variant: '#6a3c00'
  background: '#f8f9fe'
  on-background: '#191c1f'
  surface-variant: '#e0e2e7'
typography:
  display:
    fontFamily: Manrope
    fontSize: 48px
    fontWeight: '800'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  h1:
    fontFamily: Manrope
    fontSize: 32px
    fontWeight: '700'
    lineHeight: '1.3'
    letterSpacing: -0.01em
  h2:
    fontFamily: Manrope
    fontSize: 24px
    fontWeight: '700'
    lineHeight: '1.4'
  h3:
    fontFamily: Manrope
    fontSize: 20px
    fontWeight: '600'
    lineHeight: '1.4'
  body-lg:
    fontFamily: Manrope
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Manrope
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  label-md:
    fontFamily: Manrope
    fontSize: 14px
    fontWeight: '600'
    lineHeight: '1.4'
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Manrope
    fontSize: 12px
    fontWeight: '500'
    lineHeight: '1.4'
rounded:
  sm: 0.5rem
  DEFAULT: 1rem
  md: 1.5rem
  lg: 2rem
  xl: 3rem
  full: 9999px
spacing:
  base: 8px
  xs: 4px
  sm: 12px
  md: 24px
  lg: 48px
  xl: 80px
  gutter: 24px
  margin: 32px
---

## Brand & Style

This design system is built upon the "Sentinel of Softness" philosophy, merging the stoic reliability of a corporate institution with a welcoming, fluid approachability. It targets high-stakes environments—such as finance, healthcare, or enterprise infrastructure—where clarity is paramount and tension must be diffused through design.

The aesthetic follows a **Minimalist-Modern** movement. By prioritizing generous whitespace and high-contrast typography against a near-white canvas, the interface feels airy and effortless. The "resilience" is conveyed through structural stability and logical grouping, while the "flow" is realized through organic, fully rounded geometries that eliminate sharp edges, creating a safe and intuitive user experience.

## Colors

The color palette is architected to prioritize optical comfort and hierarchical clarity. The foundation is built on two soft neutrals: a primary background color that provides a crisp, clean staging area, and a secondary light grey used for structural surfaces and subtle containment.

The primary brand blue is reserved strictly for utility and emphasis—indicating primary actions, active states, and critical information. To ensure maximum accessibility, the text colors utilize a high-contrast slate scale.
- **Surface Primary (#F8FAFC):** Main application background.
- **Surface Secondary (#F1F5F9):** Inset areas, card backgrounds, and decorative elements.
- **Action Primary (#006494):** Buttons, links, and focus states.
- **Text Primary (#0F172A):** Headings and essential body copy.
- **Text Secondary (#64748B):** Metadata, helper text, and placeholders.

## Typography

This design system exclusively utilizes **Manrope**, a modern geometric sans-serif that balances technological precision with organic warmth. The typography scales are designed for rapid scanning. 

Headlines use a heavier weight and tighter letter spacing to project authority and strength. Body text maintains a generous line height (1.6) to ensure readability during long-form consumption. Label styles utilize semi-bold weights to remain legible even at smaller scales against the light background palette. Use Slate-900 for all headings to anchor the page, and Slate-500 for auxiliary body text to create a natural visual skip.

## Layout & Spacing

The layout philosophy follows a **Fixed-Grid hybrid** model. Large-screen interfaces should center content within a 1280px container, while internal components utilize a fluid 12-column system. 

The rhythm is dictated by an 8px base unit, but the "Sentinel of Softness" feel is achieved through **Extra-Large (XL) padding**. Never crowd content; allow elements to "breathe" by using 48px to 80px of vertical separation between major sections. Margins should be generous to draw the eye toward the center of the screen, reinforcing a sense of focus and calm.

## Elevation & Depth

This design system avoids heavy shadows and traditional skeuomorphism. Instead, it uses **Tonal Layering** supplemented by **Ambient Diffusion**. 

Depth is primarily created by placing White (#FFFFFF) cards on top of the Light Grey (#F1F5F9) secondary surface. When physical depth is required for interactive elements (like menus or modals), use a "Whisper Shadow": a multi-layered shadow with a large blur radius (30px+), very low opacity (4-8%), and a slight blue-grey tint to harmonize with the palette. The goal is for elements to feel like they are softly floating rather than casting a hard silhouette.

## Shapes

The shape language is the core differentiator of this design system. It utilizes **full roundness (Pill-shaped)** for nearly all interactive and structural components. 

Buttons, input fields, and tags must always use a maximum border-radius to create a "pill" effect. Larger containers like cards and modals should use a minimum of 2rem (32px) radius to maintain the soft, protective aesthetic. This total lack of sharp corners removes visual aggression, making the interface feel resilient to user error and inviting to the touch.

## Components

- **Buttons:** Primary buttons are pill-shaped, filled with Brand Blue (#006494) and white text. Secondary buttons use a thick 2px border or a soft grey ghost style.
- **Inputs:** Fields are pill-shaped with a #F1F5F9 background. Upon focus, the border transitions to Brand Blue with a soft, 4px outer glow.
- **Cards:** Cards should be white (#FFFFFF) with a 32px corner radius. Use a subtle 1px border in #F1F5F9 instead of a shadow for a cleaner, flatter appearance.
- **Chips & Tags:** Small, fully rounded pill shapes. Use light tints of the brand blue (e.g., 10% opacity) for background fills to keep them distinct but secondary.
- **Checkboxes & Radios:** These should be oversized (20px+) to accommodate the rounded aesthetic. Radios use a thick blue ring when active; checkboxes use a soft-cornered square with a 6px radius.
- **Progress Bars:** Use thick, 12px tall pill-shaped tracks with a high-contrast blue fill to emphasize "Flow."