import os
from itertools import combinations

# Create directories
os.makedirs("individual", exist_ok=True)
os.makedirs("pairwise", exist_ok=True)

# --- Individual Rankings ---
# 3 content levels (low, medium, high) x 7 page styles
for content in ["low", "medium", "high"]:
    for style in range(7):
        filename = f"individual/{content}_{style}.html"
        with open(filename, "w") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Individual Rankings - {content} - Style {style}</title>
            </head>
            <body>
                <h1>Content: {content}, Style: {style}</h1>
                <!-- Your survey content here -->
            </body>
            </html>
            """)

import os
from itertools import combinations

# Generate all 21 pairwise combinations (for 7 items)
items = ["A", "B", "C", "D", "E", "F", "G"]
all_pairs = list(combinations(items, 2))  # 21 pairs

# Generate triplets (7 triplets of 3 pairs each)
triplets = [all_pairs[i:i+3] for i in range(0, 21, 3)]

# Create GitHub Pages for each (low/medium/high)
for content in ["low", "medium", "high"]:
    os.makedirs(f"pairwise/{content}", exist_ok=True)
    for i, triplet in enumerate(triplets):
        with open(f"pairwise/{content}/triplet_{i}.html", "w") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <body>
                <h1>Pairwise Comparisons ({content})</h1>
                <p>Pairs: {triplet[0]}, {triplet[1]}, {triplet[2]}</p>
            </body>
            </html>
            """)

print("GitHub Pages generated successfully!")