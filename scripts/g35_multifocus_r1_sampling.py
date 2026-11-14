"""Fixed round-one exposure; index allocation only, no model or audio generation."""
from collections import Counter
import hashlib
import numpy as np

CENTERS = (-40, 0, 10, 30, 70)
GROUPS = tuple(f'focus_{a:+d}' for a in CENTERS) + ('background',)
QUOTAS = np.array([300] * 5 + [1500], dtype=np.int64)
BASE_NOISE_ANGLES = (-90, -60, -30, 0, 30, 60, 90)


def group_of(angle):
    matches = [i for i, center in enumerate(CENTERS) if center - 5 <= angle < center + 5]
    if len(matches) > 1:
        raise ValueError('Overlapping approved intervals')
    return matches[0] if matches else 5


def stratum(row):
    return (row['field'], row['angle'] if row['field'] == 'point' else None,
            row['noise_id'], row['snr_db'])


def original_schedule(catalog, seed, steps=3000):
    rng = np.random.default_rng(seed)
    multi = np.array([i for i, r in enumerate(catalog) if r['field'] == 'multi8'])
    point = {a: np.array([i for i, r in enumerate(catalog)
                         if r['field'] == 'point' and r['angle'] == a]) for a in BASE_NOISE_ANGLES}
    indices = []
    for _ in range(steps):
        if rng.random() < .5:
            angle = int(rng.choice(BASE_NOISE_ANGLES))
            index = int(rng.choice(point[angle]))
        else:
            index = int(rng.choice(multi))
        indices.append(index)
    return np.array(indices, dtype=np.int64), rng.bit_generator.state


def apportion(total, group_sizes):
    """Largest remainders; group order breaks ties, no fitted parameter."""
    denominator = int(sum(group_sizes))
    numerators = np.array(group_sizes, dtype=np.int64) * int(total)
    result = numerators // denominator
    order = sorted(range(len(result)), key=lambda i: (-(int(numerators[i]) % denominator), i))
    for i in order[:int(total - result.sum())]:
        result[i] += 1
    assert result.sum() == total
    return result


def categorical_transport(group_demands, class_demands, available):
    """Integral network LP, convex deviation from baseline class proportions.

    Unit arcs implement squared categorical-count deviation. This is an offline
    sampler allocation objective, never the neural training loss. The integer
    network incidence matrix gives integral vertices; require exact integers.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    gd = np.asarray(group_demands, dtype=np.int64)
    cd = np.asarray(class_demands, dtype=np.int64)
    assert gd.sum() == cd.sum()
    total = int(cd.sum())
    if total == 0:
        return np.zeros((len(gd), len(cd)), dtype=np.int64)
    costs, locations = [], []
    for g, demand in enumerate(gd):
        for j, count in enumerate(cd):
            if not available[g, j]:
                continue
            for unit in range(1, min(int(demand), int(count)) + 1):
                costs.append((2 * unit - 1) * total - 2 * int(demand) * int(count))
                locations.append((g, j))
    if not locations:
        raise ValueError('No feasible sampler support')
    row, col = [], []
    for k, (g, j) in enumerate(locations):
        row.extend((g, len(gd) + j)); col.extend((k, k))
    matrix = coo_matrix((np.ones(len(row)), (row, col)),
                        shape=(len(gd) + len(cd), len(costs))).tocsr()
    result = linprog(np.array(costs, dtype=np.float64), A_eq=matrix,
                     b_eq=np.concatenate((gd, cd)), bounds=(0, 1), method='highs')
    if not result.success:
        raise ValueError('Approved quotas cannot match baseline strata: ' + result.message)
    rounded = np.rint(result.x)
    if not np.array_equal(result.x, rounded):
        raise ValueError('Nonintegral transport solution; no tolerance relaxation')
    allocation = np.zeros((len(gd), len(cd)), dtype=np.int64)
    for value, (g, j) in zip(rounded.astype(np.int64), locations):
        allocation[g, j] += value
    assert np.array_equal(allocation.sum(axis=1), gd)
    assert np.array_equal(allocation.sum(axis=0), cd)
    assert not np.any(allocation[~available])
    return allocation


def focused_schedule(catalog, baseline_indices, seed):
    """Exact six-group quotas; match A's physical categorical tuple every step."""
    assert len(baseline_indices) == 3000 and len(catalog) == 2048
    rng = np.random.default_rng(seed + 350000)
    baseline_classes = [stratum(catalog[int(i)]) for i in baseline_indices]
    field_counts = Counter(s[0] for s in baseline_classes)
    point_quotas = apportion(field_counts['point'], QUOTAS)
    field_quotas = {'point': point_quotas, 'multi8': QUOTAS - point_quotas}
    pools = {}
    for i, r in enumerate(catalog):
        pools.setdefault((group_of(r['target_azimuth']), stratum(r)), []).append(i)
    assignments, allocation_audit = {}, []
    for field in ('point', 'multi8'):
        classes = sorted({s for s in baseline_classes if s[0] == field})
        counts = np.array([baseline_classes.count(s) for s in classes], dtype=np.int64)
        support = np.array([[(g, s) in pools for s in classes] for g in range(6)])
        allocation = categorical_transport(field_quotas[field], counts, support)
        for j, s in enumerate(classes):
            groups = np.repeat(np.arange(6), allocation[:, j])
            rng.shuffle(groups)
            assignments[s] = list(map(int, groups))
            allocation_audit.append({'stratum': list(s), 'baseline_count': int(counts[j]),
                                     'counts_by_target_group': allocation[:, j].tolist()})
    indices, groups = [], []
    for s in baseline_classes:
        g = assignments[s].pop()
        indices.append(int(rng.choice(pools[g, s]))); groups.append(g)
    indices = np.array(indices, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)
    assert all(not v for v in assignments.values())
    assert np.array_equal(np.bincount(groups, minlength=6), QUOTAS)
    assert all(stratum(catalog[int(a)]) == stratum(catalog[int(b)])
               for a, b in zip(baseline_indices, indices))
    assert all(group_of(catalog[int(i)]['target_azimuth']) == int(g) for i, g in zip(indices, groups))
    return indices, groups, {'rng_final_state': rng.bit_generator.state,
                              'field_quotas': {f: v.tolist() for f, v in field_quotas.items()},
                              'categorical_allocation': allocation_audit,
                              'global_and_per_step_strata_exact': True}


def index_sha(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def unit_checks():
    examples = {-45: 0, -35: 5, -5: 1, 0: 1, 5: 2, 15: 5,
                25: 3, 35: 5, 65: 4, 75: 5, -90: 5, 90: 5}
    assert all(group_of(x) == g for x, g in examples.items())
    allocation = categorical_transport([2, 2], [2, 2], np.ones((2, 2), dtype=bool))
    assert np.array_equal(allocation, [[1, 1], [1, 1]])
    allocation = categorical_transport([2, 2], [2, 2], np.eye(2, dtype=bool))
    assert np.array_equal(allocation, [[2, 0], [0, 2]])
    try:
        categorical_transport([1, 1], [1, 1], np.array([[False, False], [True, True]]))
    except ValueError:
        pass
    else:
        raise AssertionError('Impossible support was accepted')
    assert np.array_equal(apportion(1459, QUOTAS), [146, 146, 146, 146, 146, 729])
    return {'half_open_boundary_cases': len(examples), 'balanced_transport': True,
            'sparse_transport': True, 'expected_infeasible_fixture_rejected': True,
            'integer_apportionment': True}
