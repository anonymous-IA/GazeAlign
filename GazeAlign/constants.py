"""
Global constants shared across the GazeAlign codebase.

Keeping these in one place avoids the "magic number" duplication that
crops up when image size / patch grid are hard-coded independently in
the dataset, model, and visualization code.
"""

# ----------------------------------------------------------------------------
# Backbone / image geometry
# ----------------------------------------------------------------------------
IMG_SIZE = 518          # input resolution expected by the DINOv3 / RAD-DINO backbone
PATCH_SIZE = 14         # ViT patch size
GRID_SIZE = IMG_SIZE // PATCH_SIZE   # 37 -> 37x37 = 1369 patches
NUM_PATCHES = GRID_SIZE * GRID_SIZE

# ----------------------------------------------------------------------------
# Scanpath / fixation sequence
# ----------------------------------------------------------------------------
MAX_SCANPATH_LEN = 200   # fixations are truncated/padded to this length
SCANPATH_DIM = 3         # (x_norm, y_norm, t_norm)
TIME_COL = "Time (in secs)"
X_COL = "X_ORIGINAL"
Y_COL = "Y_ORIGINAL"
ID_COL = "DICOM_ID"

# ----------------------------------------------------------------------------
# Model dimensions
# ----------------------------------------------------------------------------
EMBED_DIM = 512          # shared projection dimension for image / scanpath embeddings
BACKBONE_HIDDEN_DIM = 768  # hidden size of the RAD-DINO / DINOv3 backbone

# ----------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_NUM_NEGATIVES = 5
