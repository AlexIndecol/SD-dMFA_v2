from __future__ import annotations

from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CRM = Path('/Users/alexcolloricchio/Desktop/CRMs')

YEARS = np.arange(1870, 2020, dtype=int)
OBS_START = 1971
BACKCAST_START = 1900
BACI_SWITCH_YEAR = 2022
TONNES_TO_KT = 1.0 / 1000.0

REGIONS = ['EU27', 'CHN', 'ROW']
MATERIALS = ['Zn', 'Ni', 'Sn']
COMMODITIES = ['concentrate', 'refined_metal', 'scrap']
END_USES = [
    'construction',
    'computers_and_precision_instruments',
    'electrical_equipment',
    'machinery_and_equipment',
    'motor_vehicles_trailers_and_semi_trailers',
    'other_transport_equipment',
    'products_nec',
]
STAGES = [
    'primary_extraction',
    'beneficiation_concentration',
    'refining_primary',
    'fabrication_and_manufacturing',
    'use_phase',
    'collection',
    'sorting_preprocessing',
    'recycling_refining_secondary',
    'residue_treatment_disposal',
    'environment',
]

EU27_ISO3 = {
    'AUT','BEL','BGR','HRV','CYP','CZE','DNK','EST','FIN','FRA','DEU','GRC','HUN','IRL','ITA',
    'LVA','LTU','LUX','MLT','NLD','POL','PRT','ROU','SVK','SVN','ESP','SWE'
}
EU27_NAMES = {
    'Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Czechia','Denmark','Estonia',
    'Finland','France','Germany','Greece','Hungary','Ireland','Italy','Latvia','Lithuania','Luxembourg',
    'Malta','Netherlands','Poland','Portugal','Romania','Slovakia','Slovak Republic','Slovenia','Spain','Sweden'
}

M_MAP = {'zinc': 'Zn', 'nickel': 'Ni', 'tin': 'Sn'}
BGS_PREFIX = {'Zn': 'Zinc', 'Ni': 'Nickel', 'Sn': 'Tin'}
OWID_ENTITY = {'Zn': 'Zinc', 'Ni': 'Nickel', 'Sn': 'Tin'}
BACI_HS22_CODES = {
    ('Zn', 'concentrate'): {'260800'},
    ('Zn', 'refined_metal'): {'790111', '790112', '790120'},
    ('Zn', 'scrap'): {'790200'},
    ('Ni', 'concentrate'): {'260400'},
    ('Ni', 'refined_metal'): {'750110', '750120', '750210', '750220'},
    ('Ni', 'scrap'): {'750300'},
    ('Sn', 'concentrate'): {'260900'},
    ('Sn', 'refined_metal'): {'800110', '800120', '800300'},
    ('Sn', 'scrap'): {'800200'},
}
UNSD_CONCORDANCE_URL = 'https://unstats.un.org/unsd/classifications/Econ/tables/HS-SITC-BEC%20Correlations_2022.xlsx'

SECTOR_TO_J = {
    'Residential_buildings': 'construction',
    'Non_residential_buildings': 'construction',
    'Roads': 'construction',
    'Civil_engineering_except_roads': 'construction',
    'Machinery_and_equipment': 'machinery_and_equipment',
    'Computers_and_precision_instruments': 'computers_and_precision_instruments',
    'Electrical_equipment': 'electrical_equipment',
    'Motor_vehicles_trailers_and_semi-trailers': 'motor_vehicles_trailers_and_semi_trailers',
    'Other_transport_equipment': 'other_transport_equipment',
    'Furniture_and_other_manufactured_goods_nec': 'products_nec',
    'Printed_matter_and_recorded_media': 'products_nec',
    'Food_packaging': 'products_nec',
    'Products_nec': 'products_nec',
}

J_TO_SECTORS = {
    'construction': [
        'Residential_buildings',
        'Non_residential_buildings',
        'Roads',
        'Civil_engineering_except_roads',
    ],
    'computers_and_precision_instruments': ['Computers_and_precision_instruments'],
    'electrical_equipment': ['Electrical_equipment'],
    'machinery_and_equipment': ['Machinery_and_equipment'],
    'motor_vehicles_trailers_and_semi_trailers': ['Motor_vehicles_trailers_and_semi-trailers'],
    'other_transport_equipment': ['Other_transport_equipment'],
    'products_nec': [
        'Furniture_and_other_manufactured_goods_nec',
        'Printed_matter_and_recorded_media',
        'Food_packaging',
        'Products_nec',
    ],
}


def map_region_iso3(iso3: str | float | None) -> str:
    if iso3 is None or (isinstance(iso3, float) and np.isnan(iso3)):
        return 'ROW'
    s = str(iso3).strip().upper()
    if s == 'CHN':
        return 'CHN'
    if s in EU27_ISO3:
        return 'EU27'
    return 'ROW'


def map_region_country_name(name: str | float | None) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return 'ROW'
    s = str(name).strip()
    if s == 'China':
        return 'CHN'
    if s in EU27_NAMES:
        return 'EU27'
    return 'ROW'


def commodity_from_subcommodity(*txts: object) -> str:
    t = ' '.join([str(x) for x in txts if x is not None and not (isinstance(x, float) and np.isnan(x))]).lower()
    if 'scrap' in t:
        return 'scrap'
    if any(k in t for k in ['ore', 'concentrate', 'mattes', 'matte', 'sinter', 'slurry']):
        return 'concentrate'
    return 'refined_metal'


def full_grid(keys: dict[str, list], value_name: str = 'value') -> pd.DataFrame:
    mi = pd.MultiIndex.from_product(keys.values(), names=list(keys.keys()))
    return mi.to_frame(index=False).assign(**{value_name: np.nan})


def reindex_time(df: pd.DataFrame, key_cols: list[str], value_col: str = 'value') -> pd.DataFrame:
    grid = full_grid({'t': YEARS.tolist(), **{k: sorted(df[k].unique().tolist()) for k in key_cols}}, value_name=value_col)
    out = grid.merge(df[['t', *key_cols, value_col]], on=['t', *key_cols], how='left')
    out = out.sort_values(['t', *key_cols]).copy()
    out[value_col] = out.groupby(key_cols)[value_col].transform(lambda s: s.interpolate(limit_direction='both'))
    out[value_col] = out.groupby(key_cols)[value_col].transform(lambda s: s.ffill().bfill())
    return out


def load_owid_world_mine() -> pd.DataFrame:
    z = CRM / 'Data' / 'ourworldindata' / 'global-mine-production-minerals.zip'
    with zipfile.ZipFile(z) as zz:
        with zz.open('global-mine-production-minerals.csv') as f:
            df = pd.read_csv(f)
    out = []
    for m, entity in OWID_ENTITY.items():
        d = df[df['Entity'] == entity][['Year', 'Global mine production of different minerals']].copy()
        d = d.rename(columns={'Year': 't', 'Global mine production of different minerals': 'value'})
        d['m'] = m
        out.append(d)
    out = pd.concat(out, ignore_index=True)
    out['t'] = out['t'].astype(int)
    out['value'] = pd.to_numeric(out['value'], errors='coerce')
    # OWID mine production is reported in metric tons; convert to kt for model inputs.
    out['value'] = out['value'] * TONNES_TO_KT
    return out.dropna(subset=['value'])


def load_bgs_production() -> tuple[pd.DataFrame, pd.DataFrame]:
    mines = []
    refines = []
    for m in MATERIALS:
        p = CRM / 'Data' / 'bgs' / f"{BGS_PREFIX[m]}_production.csv"
        df = pd.read_csv(p)
        df['t'] = pd.to_datetime(df['year'], errors='coerce').dt.year
        df = df[df['t'].between(OBS_START, 2019)]
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce') * TONNES_TO_KT
        df['r'] = df['country_iso3_code'].map(map_region_iso3)
        tbl = df['yearbook_table_trans'].fillna('').str.lower()

        mine = df[tbl.str.contains('mine production', na=False)].groupby(['t', 'r'], as_index=False)['quantity'].sum()
        mine['m'] = m
        mine = mine.rename(columns={'quantity': 'value'})
        mines.append(mine)

        refine_mask = tbl.str.contains('smelter', na=False) | tbl.str.contains('refinery', na=False) | tbl.str.contains('slab', na=False)
        ref = df[refine_mask].groupby(['t', 'r'], as_index=False)['quantity'].sum()
        ref['m'] = m
        ref = ref.rename(columns={'quantity': 'value'})
        refines.append(ref)

    mine_df = pd.concat(mines, ignore_index=True)
    refine_df = pd.concat(refines, ignore_index=True)

    for d in [mine_df, refine_df]:
        d['value'] = d['value'].fillna(0.0)
        d['value'] = d['value'].clip(lower=0.0)

    return mine_df, refine_df


def load_bgs_trade() -> tuple[pd.DataFrame, pd.DataFrame]:
    imports_all = []
    exports_all = []
    imp_files = {'Zn': 'Zinc_imports.csv', 'Ni': 'Nickel_imports.csv', 'Sn': 'Tin_metal_imports.csv'}
    exp_files = {'Zn': 'Zinc_exports.csv', 'Ni': 'Nickel_exports.csv', 'Sn': 'Tin_exports.csv'}

    for m in MATERIALS:
        imp = pd.read_csv(CRM / 'Data' / 'bgs' / imp_files[m])
        exp = pd.read_csv(CRM / 'Data' / 'bgs' / exp_files[m])

        for d, kind in [(imp, 'imports'), (exp, 'exports')]:
            d['t'] = pd.to_datetime(d['year'], errors='coerce').dt.year
            d = d[d['t'].between(OBS_START, 2019)].copy()
            d['quantity'] = pd.to_numeric(d['quantity'], errors='coerce') * TONNES_TO_KT
            d['r'] = d['country_iso3_code'].map(map_region_iso3)
            d['c'] = [
                commodity_from_subcommodity(a, b, c)
                for a, b, c in zip(
                    d.get('bgs_sub_commodity_trans', ''),
                    d.get('erml_sub_commodity', ''),
                    d.get('erml_commodity', ''),
                )
            ]
            g = d.groupby(['t', 'r', 'c'], as_index=False)['quantity'].sum()
            g['m'] = m
            g = g.rename(columns={'quantity': 'value'})
            g['value'] = g['value'].fillna(0.0).clip(lower=0.0)
            if kind == 'imports':
                imports_all.append(g)
            else:
                exports_all.append(g)

    imports_df = pd.concat(imports_all, ignore_index=True)
    exports_df = pd.concat(exports_all, ignore_index=True)

    return imports_df, exports_df


def extend_observed_with_backcast(obs: pd.DataFrame, owid_world: pd.DataFrame | None = None, ratio: float = 1.0) -> pd.DataFrame:
    mats = sorted(obs['m'].dropna().astype(str).unique().tolist())
    if not mats and owid_world is not None and 'm' in owid_world.columns:
        mats = sorted(owid_world['m'].dropna().astype(str).unique().tolist())
    if not mats:
        return pd.DataFrame(columns=['t', 'r', 'm', 'value'])

    rows = []
    for m in mats:
        d = obs[obs['m'] == m].copy()
        piv = full_grid({'t': list(range(OBS_START, 2020)), 'r': REGIONS}, '_tmp').drop(columns=['_tmp']).merge(
            d[['t', 'r', 'value']], on=['t', 'r'], how='left'
        )
        piv['value'] = piv['value'].fillna(0.0)

        # Region shares from earliest observed window.
        early = piv[piv['t'].between(OBS_START, min(OBS_START + 4, 2019))].copy()
        w = early.groupby('r', as_index=False)['value'].mean()
        if w['value'].sum() <= 0:
            w['value'] = 1.0 / len(w)
        else:
            w['value'] = w['value'] / w['value'].sum()
        share = dict(zip(w['r'], w['value']))

        world_obs = piv.groupby('t', as_index=False)['value'].sum().rename(columns={'value': 'world'})
        world = pd.DataFrame({'t': YEARS, 'world': np.nan})

        # Observed years.
        world = world.merge(world_obs, on='t', how='left', suffixes=('', '_obs'))
        world['world'] = world['world_obs']

        # Backcast with OWID if provided.
        if owid_world is not None:
            ow = owid_world[owid_world['m'] == m][['t', 'value']].rename(columns={'value': 'owid'})
            world = world.merge(ow, on='t', how='left')
            mask_1900_1970 = world['t'].between(BACKCAST_START, OBS_START - 1)
            world.loc[mask_1900_1970, 'world'] = world.loc[mask_1900_1970, 'owid'] * ratio

            # Pre-1900 hold 1900.
            world_1900 = world.loc[world['t'] == BACKCAST_START, 'world']
            if not world_1900.empty and pd.notna(world_1900.iloc[0]):
                world.loc[world['t'] < BACKCAST_START, 'world'] = float(world_1900.iloc[0])

        # Fallback hold-first if still missing.
        world['world'] = world['world'].interpolate(limit_direction='both').ffill().bfill()

        for r in REGIONS:
            rr = world[['t', 'world']].copy()
            rr['r'] = r
            rr['m'] = m
            rr['value'] = rr['world'] * share.get(r, 1.0 / len(REGIONS))
            rows.append(rr[['t', 'r', 'm', 'value']])

    out = pd.concat(rows, ignore_index=True)
    out['value'] = out['value'].fillna(0.0).clip(lower=0.0)
    return out


def extend_trade(obs: pd.DataFrame, years: np.ndarray = YEARS) -> pd.DataFrame:
    base = full_grid({'t': years.tolist(), 'r': REGIONS, 'm': MATERIALS, 'c': COMMODITIES}, '_tmp').drop(columns=['_tmp'])
    d = base.merge(obs[['t', 'r', 'm', 'c', 'value']], on=['t', 'r', 'm', 'c'], how='left')
    d = d.sort_values(['m', 'c', 'r', 't']).copy()
    d['value'] = d.groupby(['r', 'm', 'c'])['value'].transform(lambda s: s.interpolate(limit_direction='both'))
    d['value'] = d.groupby(['r', 'm', 'c'])['value'].transform(lambda s: s.ffill().bfill())
    d['value'] = d['value'].fillna(0.0).clip(lower=0.0)
    return d


def normalize_hs_code(x: object) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = int(float(s))
    except Exception:
        digits = ''.join(ch for ch in s if ch.isdigit())
        if not digits:
            return None
        v = int(digits)
    return f'{v:06d}'


def find_unsd_concordance_file() -> Path:
    candidates = [
        ROOT / 'data' / 'reference' / 'HS-SITC-BEC_Correlations_2022.xlsx',
        ROOT / 'data' / 'raw' / 'unsd' / 'HS-SITC-BEC_Correlations_2022.xlsx',
        CRM / 'Data' / 'un' / 'HS-SITC-BEC_Correlations_2022.xlsx',
        Path('/tmp/HS-SITC-BEC_Correlations_2022.xlsx'),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        'UNSD HS correlation workbook not found. Expected one of: '
        + ', '.join(str(p) for p in candidates)
    )


def load_hs92_hs22_concordance() -> pd.DataFrame:
    p = find_unsd_concordance_file()
    df = pd.read_excel(p, sheet_name='HS SITC BEC', usecols=['HS92', 'HS22'])
    df['HS92'] = df['HS92'].map(normalize_hs_code)
    df['HS22'] = df['HS22'].map(normalize_hs_code)
    df = df.dropna(subset=['HS92', 'HS22']).drop_duplicates().reset_index(drop=True)
    return df


def derive_baci_hs_baskets() -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], set[str]], pd.DataFrame]:
    conc = load_hs92_hs22_concordance()
    hs92_map: dict[tuple[str, str], set[str]] = {}
    hs22_map: dict[tuple[str, str], set[str]] = {}
    rows = []

    for key, hs22_codes in BACI_HS22_CODES.items():
        hs22_norm = {normalize_hs_code(c) for c in hs22_codes}
        hs22_norm = {c for c in hs22_norm if c is not None}
        hs92_codes = set(conc[conc['HS22'].isin(hs22_norm)]['HS92'].unique().tolist())
        if not hs92_codes:
            hs92_codes = set(hs22_norm)

        hs22_map[key] = set(hs22_norm)
        hs92_map[key] = set(hs92_codes)
        m, c = key
        for h22 in sorted(hs22_norm):
            src92 = sorted(conc.loc[conc['HS22'] == h22, 'HS92'].unique().tolist())
            if not src92:
                src92 = [h22]
            for h92 in src92:
                rows.append({'m': m, 'c': c, 'hs22': h22, 'hs92': h92})

    audit = pd.DataFrame(rows).drop_duplicates().sort_values(['m', 'c', 'hs22', 'hs92']).reset_index(drop=True)
    return hs92_map, hs22_map, audit


def load_baci_country_region_map() -> dict[int, str]:
    z = CRM / 'Data' / 'baci' / 'BACI_HS92_V202601.zip'
    with zipfile.ZipFile(z) as zz:
        with zz.open('country_codes_V202601.csv') as f:
            c = pd.read_csv(f)
    c['country_code'] = pd.to_numeric(c['country_code'], errors='coerce').astype('Int64')
    c['r'] = c['country_iso3'].map(map_region_iso3)
    c = c[c['country_code'].notna()].copy()
    return {int(k): v for k, v in zip(c['country_code'], c['r'])}


def parse_year_from_member(member: str) -> int | None:
    m = re.search(r'_Y(\d{4})_', member)
    if not m:
        return None
    return int(m.group(1))


def aggregate_baci_zip(
    zip_path: Path,
    hs_version: str,
    code_sets: dict[tuple[str, str], set[str]],
    country_to_region: dict[int, str],
) -> pd.DataFrame:
    member_pattern = f'BACI_{hs_version}_Y'
    all_codes = sorted({c for ss in code_sets.values() for c in ss})
    code_to_key = {}
    for key, ss in code_sets.items():
        for code in ss:
            code_to_key[code] = key

    rows = []
    with zipfile.ZipFile(zip_path) as zz:
        members = sorted([m for m in zz.namelist() if m.startswith(member_pattern) and m.endswith('.csv')])
        for member in members:
            year = parse_year_from_member(member)
            if year is None:
                continue

            parts = []
            with zz.open(member) as f:
                for chunk in pd.read_csv(
                    f,
                    usecols=['t', 'i', 'j', 'k', 'v'],
                    chunksize=1_000_000,
                    low_memory=False,
                ):
                    chunk['k6'] = chunk['k'].map(normalize_hs_code)
                    chunk = chunk[chunk['k6'].isin(all_codes)].copy()
                    if chunk.empty:
                        continue

                    chunk['o'] = pd.to_numeric(chunk['i'], errors='coerce').map(lambda x: country_to_region.get(int(x), 'ROW') if pd.notna(x) else 'ROW')
                    chunk['d'] = pd.to_numeric(chunk['j'], errors='coerce').map(lambda x: country_to_region.get(int(x), 'ROW') if pd.notna(x) else 'ROW')
                    chunk['value'] = pd.to_numeric(chunk['v'], errors='coerce').fillna(0.0).clip(lower=0.0)
                    chunk['key'] = chunk['k6'].map(code_to_key)
                    chunk = chunk[chunk['key'].notna()].copy()
                    if chunk.empty:
                        continue

                    chunk[['m', 'c']] = pd.DataFrame(chunk['key'].tolist(), index=chunk.index)
                    g = chunk.groupby(['t', 'o', 'd', 'm', 'c'], as_index=False)['value'].sum()
                    parts.append(g)

            if parts:
                ydf = pd.concat(parts, ignore_index=True)
                ydf = ydf.groupby(['t', 'o', 'd', 'm', 'c'], as_index=False)['value'].sum()
                rows.append(ydf)

    if not rows:
        return pd.DataFrame(columns=['t', 'o', 'd', 'm', 'c', 'value'])
    out = pd.concat(rows, ignore_index=True)
    out['t'] = pd.to_numeric(out['t'], errors='coerce').astype(int)
    out['value'] = pd.to_numeric(out['value'], errors='coerce').fillna(0.0).clip(lower=0.0)
    return out.groupby(['t', 'o', 'd', 'm', 'c'], as_index=False)['value'].sum()


def extend_bilateral_trade(obs: pd.DataFrame, years: np.ndarray) -> pd.DataFrame:
    base = full_grid({'t': years.tolist(), 'o': REGIONS, 'd': REGIONS, 'm': MATERIALS, 'c': COMMODITIES}, '_tmp').drop(columns=['_tmp'])
    d = base.merge(obs[['t', 'o', 'd', 'm', 'c', 'value']], on=['t', 'o', 'd', 'm', 'c'], how='left')
    d = d.sort_values(['m', 'c', 'o', 'd', 't']).copy()
    d['value'] = d.groupby(['o', 'd', 'm', 'c'])['value'].transform(lambda s: s.interpolate(limit_direction='both'))
    d['value'] = d.groupby(['o', 'd', 'm', 'c'])['value'].transform(lambda s: s.ffill().bfill())
    d['value'] = d['value'].fillna(0.0).clip(lower=0.0)
    return d


def bilateral_to_od_weights(bilateral: pd.DataFrame) -> pd.DataFrame:
    d = bilateral.copy()
    sums = d.groupby(['t', 'm', 'c', 'o'])['value'].transform('sum')
    d['value'] = np.where(sums > 0, d['value'] / sums, np.nan)
    d['value'] = d.groupby(['m', 'c', 'o', 'd'])['value'].transform(lambda s: s.interpolate(limit_direction='both'))
    d['value'] = d.groupby(['m', 'c', 'o', 'd'])['value'].transform(lambda s: s.ffill().bfill())
    sums2 = d.groupby(['t', 'm', 'c', 'o'])['value'].transform('sum')
    d['value'] = np.where(sums2 > 0, d['value'] / sums2, np.nan)
    d['value'] = d['value'].fillna(1.0 / len(REGIONS))
    sums3 = d.groupby(['t', 'm', 'c', 'o'])['value'].transform('sum')
    d['value'] = np.where(sums3 > 0, d['value'] / sums3, 1.0 / len(REGIONS))
    return d[['t', 'm', 'c', 'o', 'd', 'value']]


def build_gravity_od_from_marginals(imp_ext: pd.DataFrame, exp_ext: pd.DataFrame, years: np.ndarray) -> pd.DataFrame:
    od_rows = []
    imp0 = imp_ext.rename(columns={'r': 'd', 'value': 'imp'})
    exp0 = exp_ext.rename(columns={'r': 'o', 'value': 'exp'})

    for m in MATERIALS:
        for c in COMMODITIES:
            imp_mc = imp0[(imp0['m'] == m) & (imp0['c'] == c)][['t', 'd', 'imp']]
            exp_mc = exp0[(exp0['m'] == m) & (exp0['c'] == c)][['t', 'o', 'exp']]
            for t in years:
                imp_t = imp_mc[imp_mc['t'] == t].set_index('d')['imp'].to_dict()
                exp_t = exp_mc[exp_mc['t'] == t].set_index('o')['exp'].to_dict()
                for o in REGIONS:
                    raw = []
                    for d in REGIONS:
                        w = (float(exp_t.get(o, 0.0)) + 1e-9) * (float(imp_t.get(d, 0.0)) + 1e-9)
                        raw.append((d, w))
                    s = sum(w for _, w in raw)
                    if s <= 0:
                        for d, _ in raw:
                            od_rows.append({'t': int(t), 'm': m, 'c': c, 'o': o, 'd': d, 'value': 1.0 / len(REGIONS)})
                    else:
                        for d, w in raw:
                            od_rows.append({'t': int(t), 'm': m, 'c': c, 'o': o, 'd': d, 'value': w / s})
    return pd.DataFrame(od_rows)


def build_baci_linked_od_weights(years: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hs92_map, hs22_map, concord_audit = derive_baci_hs_baskets()
    country_to_region = load_baci_country_region_map()
    hs92_zip = CRM / 'Data' / 'baci' / 'BACI_HS92_V202601.zip'
    hs22_zip = CRM / 'Data' / 'baci' / 'BACI_HS22_V202601.zip'

    hs92 = aggregate_baci_zip(hs92_zip, 'HS92', hs92_map, country_to_region)
    hs22 = aggregate_baci_zip(hs22_zip, 'HS22', hs22_map, country_to_region)

    overlap = sorted(set(hs92['t'].unique()).intersection(set(hs22['t'].unique())))
    if overlap:
        chk92 = hs92[hs92['t'].isin(overlap)].groupby(['t', 'm', 'c'], as_index=False)['value'].sum().rename(columns={'value': 'hs92'})
        chk22 = hs22[hs22['t'].isin(overlap)].groupby(['t', 'm', 'c'], as_index=False)['value'].sum().rename(columns={'value': 'hs22'})
        overlap_chk = chk92.merge(chk22, on=['t', 'm', 'c'], how='outer').fillna(0.0)
        overlap_chk['rel_gap_vs_hs22'] = np.where(
            overlap_chk['hs22'] > 0,
            (overlap_chk['hs92'] - overlap_chk['hs22']) / overlap_chk['hs22'],
            np.nan,
        )
    else:
        overlap_chk = pd.DataFrame(columns=['t', 'm', 'c', 'hs92', 'hs22', 'rel_gap_vs_hs22'])

    hs92_pre = hs92[hs92['t'] < BACI_SWITCH_YEAR].copy()
    hs22_post = hs22[hs22['t'] >= BACI_SWITCH_YEAR].copy()
    linked = pd.concat([hs92_pre, hs22_post], ignore_index=True)

    linked = linked.groupby(['t', 'o', 'd', 'm', 'c'], as_index=False)['value'].sum()
    linked_ext = extend_bilateral_trade(linked, years)
    od = bilateral_to_od_weights(linked_ext)
    return od, concord_audit, overlap_chk


def load_miso_sector_shares(default_shares: dict[str, dict[str, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = CRM / 'Data' / 'MISO2_v1_lifetimes_wasteRates_recycling' / 'MISO2_v1_2_EoL_endUse.xlsx'
    df = pd.read_excel(p, sheet_name='Tabelle1')

    year_cols = [c for c in df.columns if isinstance(c, (int, np.integer)) and 2000 <= int(c) <= 2019]
    long = df.melt(id_vars=['region', 'material', 'sector'], value_vars=year_cols, var_name='t', value_name='value')
    long['t'] = long['t'].astype(int)
    long['value'] = pd.to_numeric(long['value'], errors='coerce').fillna(0.0)
    long['m'] = long['material'].str.lower().map(M_MAP)
    long = long[long['m'].isin(MATERIALS)].copy()
    long['r'] = long['region'].map(map_region_country_name)

    long['j'] = long['sector'].map(SECTOR_TO_J)
    mapped = long[long['j'].notna()].copy()

    j_vals = mapped.groupby(['t', 'r', 'm', 'j'], as_index=False)['value'].sum()
    totals = j_vals.groupby(['t', 'r', 'm'], as_index=False)['value'].sum().rename(columns={'value': 'tot'})
    j_vals = j_vals.merge(totals, on=['t', 'r', 'm'], how='left')
    j_vals['share'] = np.where(j_vals['tot'] > 0, j_vals['value'] / j_vals['tot'], np.nan)
    j_share = j_vals[['t', 'r', 'm', 'j', 'share']]

    # Expand to full historic years with hold-first/hold-last.
    grid = full_grid({'t': YEARS.tolist(), 'r': REGIONS, 'm': MATERIALS, 'j': END_USES}, 'share')
    out = grid.merge(j_share, on=['t', 'r', 'm', 'j'], how='left', suffixes=('', '_obs'))
    out['share'] = out['share_obs']
    out = out.sort_values(['r', 'm', 'j', 't'])
    out['share'] = out.groupby(['r', 'm', 'j'])['share'].transform(lambda s: s.ffill().bfill())

    # Fallback to configured defaults when still missing.
    for m in MATERIALS:
        for j in END_USES:
            mask = (out['m'] == m) & (out['j'] == j) & (out['share'].isna())
            out.loc[mask, 'share'] = float(default_shares.get(m, {}).get(j, 0.0))

    # Normalize to sum 1 per t,r,m.
    s = out.groupby(['t', 'r', 'm'])['share'].transform('sum')
    out['share'] = np.where(s > 0, out['share'] / s, 1.0 / len(END_USES))

    # Detailed sector weights for lifetime aggregation.
    sec = long[long['sector'].isin(set().union(*J_TO_SECTORS.values()))].copy()
    sec = sec.groupby(['t', 'r', 'm', 'sector'], as_index=False)['value'].sum()
    sec_avg = sec.groupby(['r', 'm', 'sector'], as_index=False)['value'].mean()
    sec_avg['j'] = sec_avg['sector'].map(SECTOR_TO_J)
    sec_avg['j_tot'] = sec_avg.groupby(['r', 'm', 'j'])['value'].transform('sum')
    sec_avg['w'] = np.where(sec_avg['j_tot'] > 0, sec_avg['value'] / sec_avg['j_tot'], np.nan)
    sec_avg['w'] = sec_avg.groupby(['r', 'm', 'j'])['w'].transform(lambda s: s.fillna(1.0 / len(s) if len(s) else np.nan))

    return out[['t', 'r', 'm', 'j', 'share']], sec_avg[['r', 'm', 'sector', 'j', 'w']]


def load_miso_lifetime(mean_or_dev: str, sec_weights: pd.DataFrame) -> pd.DataFrame:
    assert mean_or_dev in {'mean', 'sigma'}
    fn = 'MISO2_Lifetimes_v1.xlsx' if mean_or_dev == 'mean' else 'MISO2_Lifetimes_deviation_v1.xlsx'
    p = CRM / 'Data' / 'MISO2_v1_lifetimes_wasteRates_recycling' / fn

    cols = pd.read_excel(p, sheet_name='values', nrows=0).columns.tolist()
    year_cols = [c for c in cols if isinstance(c, (int, np.integer)) and 2000 <= int(c) <= 2016]
    usecols = [0, 2, 3] + [cols.index(c) for c in year_cols]

    df = pd.read_excel(p, sheet_name='values', usecols=usecols)
    df.columns = ['country', 'material', 'sector', *year_cols]
    df['m'] = df['material'].str.lower().map(M_MAP)
    df = df[df['m'].isin(MATERIALS)].copy()
    df['r'] = df['country'].map(map_region_country_name)
    df['x'] = df[year_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    sec = df.groupby(['r', 'm', 'sector'], as_index=False)['x'].mean()
    sec = sec[sec['sector'].isin(set().union(*J_TO_SECTORS.values()))].copy()

    # Weighted aggregation of detailed sectors to model end-uses.
    j_rows = []
    for r in REGIONS:
        for m in MATERIALS:
            ss = sec[(sec['r'] == r) & (sec['m'] == m)]
            ww = sec_weights[(sec_weights['r'] == r) & (sec_weights['m'] == m)]
            for j in END_USES:
                sectors = J_TO_SECTORS[j]
                s = ss[ss['sector'].isin(sectors)][['sector', 'x']]
                w = ww[ww['sector'].isin(sectors)][['sector', 'w']]
                z = s.merge(w, on='sector', how='left')
                if z.empty:
                    val = np.nan
                else:
                    z['w'] = z['w'].fillna(1.0 / len(z))
                    z['w'] = z['w'] / z['w'].sum()
                    val = float((z['x'] * z['w']).sum())
                j_rows.append({'r': r, 'm': m, 'j': j, 'value': val})

    out = pd.DataFrame(j_rows)
    if mean_or_dev == 'mean':
        out['value'] = out['value'].clip(lower=0.5)
    else:
        # Interpret MISO deviation as lognormal sigma proxy.
        out['value'] = out['value'].clip(lower=0.05, upper=1.5)
    return out


def apply_template_fill(path: Path, key_cols: list[str], values: pd.DataFrame) -> None:
    tpl = pd.read_csv(path)
    values = values.copy()
    for c in key_cols:
        if c in tpl.columns and c in values.columns:
            if c == 't':
                tpl[c] = pd.to_numeric(tpl[c], errors='coerce').astype('Int64')
                values[c] = pd.to_numeric(values[c], errors='coerce').astype('Int64')
            else:
                tpl[c] = tpl[c].astype(str)
                values[c] = values[c].astype(str)

    tpl = tpl.drop_duplicates(subset=key_cols, keep='first')
    values = values.drop_duplicates(subset=key_cols, keep='last')

    out = tpl.merge(values[key_cols + ['value']], on=key_cols, how='left', suffixes=('', '_new'))
    out['value'] = pd.to_numeric(out['value_new'], errors='coerce').combine_first(pd.to_numeric(out['value'], errors='coerce'))
    out = out[tpl.columns]
    out.to_csv(path, index=False)


def main() -> None:
    # Defaults for share fallback from SD parameters.
    p_sd = yaml.safe_load((ROOT / 'configs' / 'parameters' / 'parameters_sd.yml').read_text())['sd']
    share_defaults_raw = p_sd['demand']['end_use_share_default_by_material']
    share_defaults = {m: {j: float(v) for j, v in d.items()} for m, d in share_defaults_raw.items()}

    # BGS observed series.
    mine_obs, refine_obs = load_bgs_production()
    imp_obs, exp_obs = load_bgs_trade()

    # OWID world mine for backcasting.
    owid = load_owid_world_mine()

    # Ratios for backcast scaling.
    mine_world = mine_obs.groupby(['t', 'm'], as_index=False)['value'].sum().rename(columns={'value': 'mine'})
    ref_world = refine_obs.groupby(['t', 'm'], as_index=False)['value'].sum().rename(columns={'value': 'ref'})

    imp_ref = imp_obs[imp_obs['c'] == 'refined_metal'].groupby(['t', 'm'], as_index=False)['value'].sum().rename(columns={'value': 'imp_ref'})
    exp_ref = exp_obs[exp_obs['c'] == 'refined_metal'].groupby(['t', 'm'], as_index=False)['value'].sum().rename(columns={'value': 'exp_ref'})
    gas_world = mine_world.merge(ref_world, on=['t', 'm'], how='left').merge(imp_ref, on=['t', 'm'], how='left').merge(exp_ref, on=['t', 'm'], how='left')
    gas_world[['ref', 'imp_ref', 'exp_ref']] = gas_world[['ref', 'imp_ref', 'exp_ref']].fillna(0.0)
    gas_world['gas'] = (gas_world['ref'] + gas_world['imp_ref'] - gas_world['exp_ref']).clip(lower=0.0)

    ratios = []
    for m in MATERIALS:
        d = gas_world[(gas_world['m'] == m) & (gas_world['t'].between(1971, 1980))].merge(
            mine_world[(mine_world['m'] == m) & (mine_world['t'].between(1971, 1980))],
            on=['t', 'm'],
            suffixes=('_g', '_mine'),
        )
        denom = d['mine_mine'].replace(0, np.nan)
        gas_ratio = float((d['gas'] / denom).median()) if len(d) else 1.0
        ref_ratio = float((d['ref'] / denom).median()) if len(d) else 0.8
        if not np.isfinite(gas_ratio):
            gas_ratio = 1.0
        if not np.isfinite(ref_ratio):
            ref_ratio = 0.8
        ratios.append({'m': m, 'gas_ratio': max(gas_ratio, 0.01), 'ref_ratio': max(ref_ratio, 0.01)})
    ratios = pd.DataFrame(ratios)

    # Extend mine and refine with OWID backcast.
    mine_ext_parts = []
    refine_ext_parts = []
    for m in MATERIALS:
        ow = owid[owid['m'] == m]
        gas_ratio = float(ratios.loc[ratios['m'] == m, 'gas_ratio'].iloc[0])
        ref_ratio = float(ratios.loc[ratios['m'] == m, 'ref_ratio'].iloc[0])

        mine_m = extend_observed_with_backcast(mine_obs[mine_obs['m'] == m], owid_world=ow, ratio=1.0)
        ref_m = extend_observed_with_backcast(refine_obs[refine_obs['m'] == m], owid_world=ow, ratio=ref_ratio)
        mine_ext_parts.append(mine_m)
        refine_ext_parts.append(ref_m)

    mine_ext = pd.concat(mine_ext_parts, ignore_index=True)
    refine_ext = pd.concat(refine_ext_parts, ignore_index=True)

    # Extend trade (hold/interpolate by region/material/commodity).
    imp_ext = extend_trade(imp_obs, years=YEARS)
    exp_ext = extend_trade(exp_obs, years=YEARS)

    # Gas total by region/material/year.
    imp_ref_ext = imp_ext[imp_ext['c'] == 'refined_metal'][['t', 'r', 'm', 'value']].rename(columns={'value': 'imp_ref'})
    exp_ref_ext = exp_ext[exp_ext['c'] == 'refined_metal'][['t', 'r', 'm', 'value']].rename(columns={'value': 'exp_ref'})
    gas_total = full_grid({'t': YEARS.tolist(), 'r': REGIONS, 'm': MATERIALS}, 'gas')
    gas_total = gas_total.merge(refine_ext.rename(columns={'value': 'ref'}), on=['t', 'r', 'm'], how='left')
    gas_total = gas_total.merge(imp_ref_ext, on=['t', 'r', 'm'], how='left')
    gas_total = gas_total.merge(exp_ref_ext, on=['t', 'r', 'm'], how='left')
    gas_total[['ref', 'imp_ref', 'exp_ref']] = gas_total[['ref', 'imp_ref', 'exp_ref']].fillna(0.0)
    gas_total['gas'] = (gas_total['ref'] + gas_total['imp_ref'] - gas_total['exp_ref']).clip(lower=0.0)

    # Sector shares from MISO EoL dataset.
    j_shares, sec_weights = load_miso_sector_shares(share_defaults)

    # Lifetime mean/sigma and mu.
    life_mean = load_miso_lifetime('mean', sec_weights)
    life_sigma = load_miso_lifetime('sigma', sec_weights)
    life = life_mean.merge(life_sigma, on=['r', 'm', 'j'], suffixes=('_mean', '_sigma'))
    life['value_sigma'] = life['value_sigma'].fillna(0.30)
    life['value_mean'] = life['value_mean'].fillna(20.0).clip(lower=0.5)
    life['mu'] = np.log(life['value_mean']) - 0.5 * np.square(life['value_sigma'])

    # GAS to use by end-use.
    gas_j = j_shares.merge(gas_total[['t', 'r', 'm', 'gas']], on=['t', 'r', 'm'], how='left')
    gas_j['gas'] = gas_j['gas'].fillna(0.0)
    gas_j['value'] = gas_j['share'] * gas_j['gas']
    gas_j = gas_j[['t', 'r', 'm', 'j', 'value']]

    # In-use stock proxy from dynamic stock identity with average lifetime.
    L = life_mean.rename(columns={'value': 'L'})[['r', 'm', 'j', 'L']]
    stock = gas_j.merge(L, on=['r', 'm', 'j'], how='left').rename(columns={'value': 'inflow'})
    stock['L'] = stock['L'].fillna(20.0).clip(lower=0.5)
    stock = stock.sort_values(['r', 'm', 'j', 't']).copy()
    stock['value'] = 0.0
    for (r, m, j), g in stock.groupby(['r', 'm', 'j'], sort=False):
        idx = g.index.tolist()
        prev = None
        Lval = float(g['L'].iloc[0])
        for k in idx:
            inflow = float(g.at[k, 'inflow'])
            if prev is None:
                cur = inflow * Lval
            else:
                cur = prev + inflow - prev / Lval
            if cur < 0:
                cur = 0.0
            stock.at[k, 'value'] = cur
            prev = cur
    in_use = stock[['t', 'r', 'm', 'j', 'value']].copy()

    # OD weights: BACI (HS92 linked to HS22 via official UNSD concordance), with explicit
    # fallback to gravity-style marginals for years before BACI coverage.
    od_template = ROOT / 'data' / 'exogenous' / 'od_preference_weight_0_1.csv'
    od_years = np.sort(pd.to_numeric(pd.read_csv(od_template, usecols=['t'])['t'], errors='coerce').dropna().astype(int).unique())
    imp_ext_od = extend_trade(imp_obs, years=od_years)
    exp_ext_od = extend_trade(exp_obs, years=od_years)
    od_fallback = build_gravity_od_from_marginals(imp_ext_od, exp_ext_od, years=od_years)
    od_baci, concord_audit, baci_overlap_chk = build_baci_linked_od_weights(years=od_years)
    od = od_fallback.merge(
        od_baci.rename(columns={'value': 'value_baci'}),
        on=['t', 'm', 'c', 'o', 'd'],
        how='left',
    )
    od['value'] = np.where(
        (od['t'] >= 1995) & od['value_baci'].notna(),
        od['value_baci'],
        od['value'],
    )
    od = od[['t', 'm', 'c', 'o', 'd', 'value']]

    # Capacity stage observed.
    rates_coll = p_sd['r_strategies']['baseline_targets_2020']['collection_rate_frac']
    rates_rr = p_sd['r_strategies']['baseline_targets_2020']['eol_recycling_rate_frac']

    cap = full_grid({'t': YEARS.tolist(), 'r': REGIONS, 'm': MATERIALS, 'p': STAGES}, 'value')
    cap = cap.merge(mine_ext.rename(columns={'value': 'mine'}), on=['t', 'r', 'm'], how='left')
    cap = cap.merge(refine_ext.rename(columns={'value': 'ref'}), on=['t', 'r', 'm'], how='left')
    cap = cap.merge(gas_total[['t', 'r', 'm', 'gas']], on=['t', 'r', 'm'], how='left')
    cap[['mine', 'ref', 'gas']] = cap[['mine', 'ref', 'gas']].fillna(0.0)

    cap['collection_rate'] = cap['r'].map(lambda x: float(rates_coll.get(x, 0.5)))
    cap['eol_rr'] = cap.apply(lambda x: float((rates_rr.get(x['r'], {}) or {}).get(x['m'], 0.4)), axis=1)

    cap['value'] = 0.0
    cap.loc[cap['p'] == 'primary_extraction', 'value'] = cap.loc[cap['p'] == 'primary_extraction', 'mine']
    cap.loc[cap['p'] == 'beneficiation_concentration', 'value'] = cap.loc[cap['p'] == 'beneficiation_concentration', 'mine']
    cap.loc[cap['p'] == 'refining_primary', 'value'] = cap.loc[cap['p'] == 'refining_primary', 'ref']
    cap.loc[cap['p'] == 'fabrication_and_manufacturing', 'value'] = cap.loc[cap['p'] == 'fabrication_and_manufacturing', 'gas']
    cap.loc[cap['p'] == 'use_phase', 'value'] = cap.loc[cap['p'] == 'use_phase', 'gas']

    coll = (cap['gas'] * cap['collection_rate']).clip(lower=0.0)
    recy = (cap['gas'] * cap['eol_rr']).clip(lower=0.0)
    recy = np.minimum(recy, coll)

    cap.loc[cap['p'] == 'collection', 'value'] = coll[cap['p'] == 'collection']
    cap.loc[cap['p'] == 'sorting_preprocessing', 'value'] = (coll * 0.90)[cap['p'] == 'sorting_preprocessing']
    cap.loc[cap['p'] == 'recycling_refining_secondary', 'value'] = recy[cap['p'] == 'recycling_refining_secondary']
    residue = (coll - recy).clip(lower=0.0)
    cap.loc[cap['p'] == 'residue_treatment_disposal', 'value'] = residue[cap['p'] == 'residue_treatment_disposal']
    cap.loc[cap['p'] == 'environment', 'value'] = (residue * 0.50)[cap['p'] == 'environment']
    cap['value'] = cap['value'].fillna(0.0).clip(lower=0.0)

    # Internal consistency checks.
    chk = gas_j.groupby(['t', 'r', 'm'], as_index=False)['value'].sum().rename(columns={'value': 'gas_j_sum'})
    chk = chk.merge(gas_total[['t', 'r', 'm', 'gas']], on=['t', 'r', 'm'], how='left')
    max_abs_gap = float((chk['gas_j_sum'] - chk['gas']).abs().max())

    wsum = od.groupby(['t', 'm', 'c', 'o'], as_index=False)['value'].sum()
    max_wsum_err = float((wsum['value'] - 1.0).abs().max())

    # Cross-check BGS world mine vs OWID in overlap.
    bgs_world = mine_obs.groupby(['t', 'm'], as_index=False)['value'].sum().rename(columns={'value': 'bgs'})
    comp = bgs_world.merge(owid[['t', 'm', 'value']].rename(columns={'value': 'owid'}), on=['t', 'm'], how='inner')
    comp = comp[comp['t'].between(1971, 2019)]
    comp['rel_err'] = (comp['bgs'] - comp['owid']).abs() / comp['owid'].replace(0, np.nan)
    med_rel = comp.groupby('m', as_index=False)['rel_err'].median().rename(columns={'rel_err': 'median_rel_err'})

    baci_gap = baci_overlap_chk.copy()
    baci_gap_med = (
        baci_gap.groupby(['m', 'c'], as_index=False)['rel_gap_vs_hs22']
        .median()
        .rename(columns={'rel_gap_vs_hs22': 'median_rel_gap_vs_hs22'})
        if not baci_gap.empty
        else pd.DataFrame(columns=['m', 'c', 'median_rel_gap_vs_hs22'])
    )
    audit_dir = ROOT / 'data' / 'reference'
    audit_dir.mkdir(parents=True, exist_ok=True)
    concord_audit.to_csv(audit_dir / 'baci_hs22_to_hs92_concordance_used.csv', index=False)
    baci_gap.to_csv(audit_dir / 'baci_hs92_vs_hs22_overlap_diagnostics.csv', index=False)

    print('CHECK: max |sum_j gas_j - gas_total| =', max_abs_gap)
    print('CHECK: max |sum_d od_weight - 1| =', max_wsum_err)
    print('CHECK: median relative error BGS mine vs OWID (1971-2019):')
    print(med_rel.to_string(index=False))
    print('CHECK: BACI concordance pairs used (HS22 target -> HS92 source):', len(concord_audit))
    print('CHECK: BACI HS92-vs-HS22 overlap median relative gap by material/commodity:')
    if len(baci_gap_med):
        print(baci_gap_med.to_string(index=False))
    else:
        print('No BACI HS92/HS22 overlap years available.')

    # Fill templates.
    apply_template_fill(ROOT / 'data' / 'exogenous' / 'gas_to_use_observed_kt_per_yr.csv', ['t', 'r', 'm', 'j'], gas_j)
    apply_template_fill(ROOT / 'data' / 'exogenous' / 'in_use_stock_observed_kt.csv', ['t', 'r', 'm', 'j'], in_use)
    apply_template_fill(ROOT / 'data' / 'exogenous' / 'od_preference_weight_0_1.csv', ['t', 'm', 'c', 'o', 'd'], od)
    apply_template_fill(ROOT / 'data' / 'exogenous' / 'capacity_stage_observed_kt_per_yr.csv', ['t', 'r', 'm', 'p'], cap[['t', 'r', 'm', 'p', 'value']])
    apply_template_fill(ROOT / 'data' / 'exogenous' / 'lifetime_lognormal_sigma.csv', ['r', 'm', 'j'], life_sigma)
    apply_template_fill(ROOT / 'data' / 'exogenous' / 'lifetime_lognormal_mu.csv', ['r', 'm', 'j'], life[['r', 'm', 'j', 'mu']].rename(columns={'mu': 'value'}))

    # Keep lifetime_mean_years consistent by filling missing construction values from MISO means.
    apply_template_fill(ROOT / 'data' / 'exogenous' / 'lifetime_mean_years.csv', ['r', 'm', 'j'], life_mean)

    print('Wrote required baseline exogenous files.')


if __name__ == '__main__':
    main()
