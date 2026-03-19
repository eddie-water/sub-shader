"""
Create a showcase of 20 diverse fonts - serif, sans-serif, and monospace.
Shows what's actually available on the system.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Get all available fonts
all_fonts = {}
for font in fm.fontManager.ttflist:
    if font.name not in all_fonts:
        all_fonts[font.name] = font.fname

# Curate a list of 20 diverse fonts - prefer actual available ones
desired_fonts = [
    # Serifs (6)
    'cmr10',                    # Computer Modern Roman - thin, elegant, mathematical
    'DejaVu Serif',             # Clean serif
    'DejaVu Serif Display',     # Serif display variant
    'cmb10',                    # Computer Modern Bold
    'cmtt10',                   # Computer Modern Typewriter
    'cmss10',                   # Computer Modern Sans

    # Sans-serif (8)
    'DejaVu Sans',              # Clean, neutral sans-serif
    'DejaVu Sans Display',      # Sans display variant
    'DejaVu Sans Mono',         # Sans monospace
    'Ubuntu Sans',              # Modern sans-serif
    'Ubuntu Sans Mono',         # Ubuntu monospace
    'DejaVu Math TeX Gyre',     # Math-optimized
    'cmmi10',                   # Computer Modern Math Italics
    'cmex10',                   # Computer Modern Math Extensions

    # Backup: DejaVu Sans repeated if needed (6)
    'DejaVu Sans',
    'DejaVu Sans',
    'DejaVu Sans',
    'DejaVu Sans',
    'DejaVu Sans',
    'DejaVu Sans',
]

# Filter to only available fonts
available_fonts = []
for font_name in desired_fonts:
    if font_name in all_fonts:
        if font_name not in [f for f in available_fonts]:
            available_fonts.append(font_name)

# If we have fewer than 20, add more DejaVu variants
while len(available_fonts) < 20:
    for font_name in sorted(all_fonts.keys()):
        if font_name not in available_fonts and len(available_fonts) < 20:
            available_fonts.append(font_name)

print(f"Showing {len(available_fonts)} available fonts:\n")

# Create showcase figure (4 columns, 5 rows)
fig = plt.figure(figsize=(16, 14))
fig.suptitle('20 Font Showcase - Benchmark Figure Font Selection', fontsize=20, fontweight='bold', y=0.995)

for idx, font_name in enumerate(available_fonts[:20]):
    ax = fig.add_subplot(5, 4, idx + 1)

    # Determine if serif or sans
    if 'serif' in font_name.lower():
        family = 'serif'
        matplotlib.rcParams['font.serif'] = [font_name]
    else:
        family = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [font_name]

    matplotlib.rcParams['font.family'] = family

    # Render samples
    ax.text(0.5, 0.7, font_name, fontsize=12, ha='center', weight='bold', family=family)
    ax.text(0.5, 0.45, 'STFT | PyWavelet', fontsize=9, ha='center', family=family)
    ax.text(0.5, 0.2, '1.43 ms/frame', fontsize=8, ha='center', family=family)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Add background to distinguish them
    ax.set_facecolor('#f8f8f8' if idx % 2 == 0 else '#ffffff')

    print(f"{idx+1:2d}. {font_name}")

plt.tight_layout()
plt.savefig('assets/images/font_showcase_20.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved -> assets/images/font_showcase_20.png")
