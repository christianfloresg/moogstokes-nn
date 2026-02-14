import os
import re
import shutil

def remove_vsin_duplicates(base_path):
    """
    Moves folders with vsin ending in .0 into 'trash' if they are duplicates
    of existing folders without the trailing .0.
    """
    trash_path = os.path.join(base_path, "trash")
    os.makedirs(trash_path, exist_ok=True)

    # Regex to extract parameters
    pattern = re.compile(
        r"T(\d+)_G(\d+)_Veil([\d.]+)_Bf([\d.]+)_vsin([\d.]+)"
    )

    seen = set()

    # First pass: collect original folders (without trailing .0 in vsin)
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path) or folder_name == "trash":
            continue

        match = pattern.search(folder_name)
        if match:
            vsin_val = match.group(5)
            if not vsin_val.endswith(".0"):
                key = (
                    int(match.group(1)),         # T
                    int(match.group(2)),         # G
                    float(match.group(3)),       # Veil
                    float(match.group(4)),       # Bf
                    float(vsin_val)              # vsin
                )
                seen.add(key)

    # Second pass: move duplicates with trailing .0
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path) or folder_name == "trash":
            continue

        match = pattern.search(folder_name)
        if match:
            vsin_val = match.group(5)
            if vsin_val.endswith(".0"):
                key = (
                    int(match.group(1)),         # T
                    int(match.group(2)),         # G
                    float(match.group(3)),       # Veil
                    float(match.group(4)),       # Bf
                    float(vsin_val)              # vsin
                )
                if key in seen:
                    dest_path = os.path.join(trash_path, folder_name)
                    print(f"Moving duplicate with trailing .0 to trash: {folder_name}")
                    shutil.move(folder_path, dest_path)

def rename_folders(base_path):
    """
    Renames folders like:
    iSHELL_0.75K2_T3000_G250_Veil0.0_Bf0.0_vsin12
    to
    iSHELL_0.75K2_T3000_G2500_Veil0.000_Bf0.000_vsin12.000
    """
    # Regex to match the folder pattern
    pattern = re.compile(
        r"(iSHELL_[\d.]+K\d+_T)(\d+)(_G)(\d+)(_Veil)([\d.]+)(_Bf)([\d.]+)(_vsin)([\d.]+)"
    )

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        match = pattern.match(folder_name)
        if match:
            new_name = (
                f"{match.group(1)}{match.group(2)}"          # T stays same
                f"{match.group(3)}{match.group(4)}0"         # G gets extra zero
                f"{match.group(5)}{float(match.group(6)):.3f}" # Veil to 3 decimals
                f"{match.group(7)}{float(match.group(8)):.3f}" # Bf to 3 decimals
                f"{match.group(9)}{float(match.group(10)):.3f}" # vsin to 3 decimals
            )
            new_path = os.path.join(base_path, new_name)
            print(f"Renaming:\n  {folder_name}\n  -> {new_name}\n")
            os.rename(folder_path, new_path)

# #!/usr/bin/env python3
# import sys
# import tempfile
# import os

# def scale_first_column_inplace(filename, factor=1e4):
#     # Write to a temporary file, then atomically replace the original
#     with open(filename, "r") as fin, tempfile.NamedTemporaryFile(
#         "w", delete=False, dir=os.path.dirname(os.path.abspath(filename))
#     ) as fout:
#         temp_name = fout.name
#         for line in fin:
#             stripped = line.strip()
#             if not stripped:
#                 fout.write(line)
#                 continue

#             parts = stripped.split()
#             try:
#                 parts[0] = f"{float(parts[0]) * factor:.16g}"
#                 fout.write(" ".join(parts) + "\n")
#             except ValueError:
#                 # Non-numeric line: write unchanged
#                 fout.write(line)

#     os.replace(temp_name, filename)

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         sys.exit(f"Usage: {sys.argv[0]} input_file")

#     scale_first_column_inplace(sys.argv[1])


if __name__ == "__main__":
    base_path = "data/moog-stokes"  # Replace with your folder path
    #base_path = "/Users/ellenlee/Downloads/models-for-proplyds"
    #remove_vsin_duplicates(base_path)
    #rename_folders(base_path)
