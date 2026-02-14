import os
import re
import itertools
from collections import defaultdict

FOLDER_PATTERN = re.compile(
    r'T(?P<T>\d+)_G(?P<G>\d+)_Veil[\d\.]+_Bf(?P<Bf>[\d\.]+)_vsin(?P<vsin>[\d\.]+)'
)

PARAM_ORDER = ['T', 'G', 'Bf', 'vsin']

def filter_missing_by_param(missing, param_index, param_value):
    """
    Filters the missing combinations by the given parameter index (0=T, 1=G, etc.) and value.
    """
    return [x for x in missing if x[param_index] == param_value]

def print_missing_for_param(path, param_name, value):
    param_indices = {'T': 0, 'G': 1, 'Bf': 2, 'vsin': 3}
    param_index = param_indices[param_name]
    folders = find_all_folders(path)
    params = [parse_parameters_from_foldername(name) for name in folders]
    params = [p for p in params if p is not None]
    axes_vals = [sorted(set(p[i] for p in params)) for i in range(4)]
    full_grid = build_full_grid(axes_vals)
    existing = set(params)
    missing = full_grid - existing
    missing_tuples = [tuple(x) for x in missing]
    filtered_missing = filter_missing_by_param(missing_tuples, param_index, value)
    print(f"\nMissing models for {param_name}={value}:")
    if not filtered_missing:
        print(f"No missing combinations for {param_name}={value}.")
        return
    for line in compact_grid(filtered_missing, axes_vals, PARAM_ORDER):
        print(line)


def parse_parameters_from_foldername(name):
    m = FOLDER_PATTERN.search(name)
    if not m:
        return None
    return (
        int(m.group('T')),
        int(m.group('G')),
        float(m.group('Bf')),
        float(m.group('vsin')),
    )

def find_all_folders(path='.'):
    return [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name)) and name.startswith('iSHELL')
    ]

def build_full_grid(param_values):
    return set(itertools.product(*param_values))

def is_arithmetic_progression(vals):
    vals = sorted(vals)
    if len(vals) < 2:
        return False, None
    step = round(vals[1] - vals[0], 10)
    for i in range(2, len(vals)):
        if round(vals[i] - vals[i-1], 10) != step:
            return False, None
    return True, step

def format_vals(vals, label):
    # Defensive: flatten if tuples (shouldn't happen, but if so take first value)
    scalars = [v[0] if isinstance(v, tuple) else v for v in vals]
    scalars = sorted(set(scalars))
    if len(scalars) == 1:
        return f"{label}={scalars[0]}"
    is_ap, step = is_arithmetic_progression(scalars)
    if is_ap:
        return f"{label} from {scalars[0]} to {scalars[-1]} (step {step})"
    else:
        return f"{label} = {', '.join(str(v) for v in scalars)}"

def compact_grid(missing, axes_vals, labels, depth=0, prefix=None):
    """
    Bottom-up recursion: At each level, group by current axis.
    If all subgroups are full, print as a range. Else, go deeper.
    """
    if prefix is None:
        prefix = []
    n_axes = len(labels)

    # Base case: only one axis left, print as a range
    if depth == n_axes - 1:
        # At this point missing should be a flat list of values, not tuples
        scalars = [m[0] if isinstance(m, tuple) else m for m in missing]
        line = ', '.join([f"{labels[i]}={prefix[i]}" for i in range(len(prefix))] + [format_vals(scalars, labels[depth])])
        yield line
        return

    # Group by current axis value
    groups = defaultdict(list)
    for tup in missing:
        # Defensive: if not a tuple, wrap in tuple (should not be needed but safe)
        first = tup[0] if isinstance(tup, tuple) else tup
        rest = tup[1:] if isinstance(tup, tuple) else ()
        groups[first].append(rest)

    all_axis_vals = axes_vals[depth]
    group_keys = sorted(groups.keys())

    # Check if this level can be compacted
    can_compact = set(group_keys) == set(all_axis_vals)
    for key in group_keys:
        sub_missing = groups[key]
        if depth+1 == n_axes-1:
            sub_vals = [x[0] if isinstance(x, tuple) else x for x in sub_missing]
            if set(sub_vals) != set(axes_vals[depth+1]):
                can_compact = False
                break
        else:
            sub_list = list(compact_grid(sub_missing, axes_vals, labels, depth+1, prefix + [key]))
            if len(sub_list) > 1:
                can_compact = False
                break

    if can_compact:
        fields = [f"{labels[i]}={prefix[i]}" for i in range(len(prefix))] if prefix else []
        fields.append(format_vals(group_keys, labels[depth]))
        for d in range(depth+1, n_axes):
            fields.append(format_vals(axes_vals[d], labels[d]))
        yield ', '.join(fields)
    else:
        for key in group_keys:
            for line in compact_grid(groups[key], axes_vals, labels, depth+1, prefix + [key]):
                yield line

def main(path='.'):
    print(f"Scanning path: {path}")
    folders = find_all_folders(path)
    print(f"Total folders found: {len(folders)}")
    params = [parse_parameters_from_foldername(name) for name in folders]
    params = [p for p in params if p is not None]

    # Gather unique values for each axis
    axes_vals = [sorted(set(p[i] for p in params)) for i in range(4)]

    print("Unique values:")
    for label, vals in zip(PARAM_ORDER, axes_vals):
        print(f"  {label}: {vals}")

    full_grid = build_full_grid(axes_vals)
    existing = set(params)
    missing = full_grid - existing
    print(f"\nMissing {len(missing)} model(s).")

    if not missing:
        print("Grid is complete! No missing combinations.")
        return

    print("\nMissing models summary (compacted):")
    missing_tuples = [tuple(x) for x in missing]
    for line in compact_grid(missing_tuples, axes_vals, PARAM_ORDER):
        print(line)


if __name__ == "__main__":
    # main(path='data/moog-stokes')
    print_missing_for_param('data/moog-stokes', 'G', 200)
