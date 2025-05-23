#!/usr/bin/env python
# coding: utf-8

# ## 1. Setup

# ### 1.1 Imports
# 
# Here we pull in all standard and third-party libraries used across the notebook.
# 

# In[123]:


import os
import re
import subprocess
from tqdm import tqdm
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from collections import defaultdict
from git import Repo, BadName
from pathlib import Path
import json
import requests
import urllib.parse
from datetime import datetime
from packaging.version import Version, InvalidVersion
import pandas as pd
import tomli
import ast
import fnmatch
import copy


# ### 1.2 Global Variables
# 
# Initialize caches, default parameters, and any constants.

# In[124]:


# The repository to analyze
import sys


repo_name = sys.argv[1]
repo_url = "https://github.com/" + repo_name
destination_path = "./" + repo_name
repo_path = repo_name
prefix = repo_name.split("/")[0] + "_" + repo_name.split("/")[1]



# The caches for python/js, maven, and gradle
file_cache = {}
properties_cache = {}
NAMESPACE = {'mvn': 'http://maven.apache.org/POM/4.0.0'}
properties = {}
file_cache_python = {}



# Caches for the versions
commits_with_date = {}
cached_versions = defaultdict(dict)


# ## 2. Function Definitions
# 

# 
# ### 2.1 Compute Metrics per Commit

# In[125]:


def compute_per_commit_file_metrics(json_path, weeks=13):
    df = pd.read_json(json_path)
    if df.empty:
        return pd.DataFrame()
    df['commit_date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[
        df['change_type'].isin(['added', 'updated']) &
        df['commit_date'].notna()
    ].rename(columns={'filename': 'file_path', 'commit': 'commit_hash'})

    for col in ['latency', 'proj_version_lag', 'criticality']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    start = df['commit_date'].min().normalize()
    end   = df['commit_date'].max().normalize() + pd.Timedelta(weeks=weeks)
    bins  = pd.date_range(start, end, freq=f'{weeks}W')
    df['interval_start'] = pd.cut(df['commit_date'], bins=bins, labels=bins[:-1])

    unique_per_quarter = (
        df
        .groupby('interval_start', observed=False)['dependency']
        .nunique()
        .reset_index(name='proj_update_frequency')
    )

    dep_intervals = (
        df
        .sort_values(['dependency', 'commit_date'])
        .groupby('dependency')['commit_date']
        .apply(lambda x: x.diff().dt.total_seconds().dropna() / (3600 * 24))
        .reset_index(name='delta_days')
    )
    avg_interval = (
        dep_intervals
        .groupby('dependency')['delta_days']
        .mean()
        .reset_index(name='avg_update_interval_days')
    )
    avg_interval['avg_update_interval_weeks'] = avg_interval['avg_update_interval_days'] / 7

    df = (
        df
        .merge(unique_per_quarter, on='interval_start', how='left')
        .merge(avg_interval[['dependency','avg_update_interval_weeks']], on='dependency', how='left')
    )



    agg = (
        df
        .groupby(['commit_hash','commit_date','file_path'])
        .agg(
            proj_update_frequency      = ('proj_update_frequency',     'first'),
            proj_mtbu                  = ('avg_update_interval_weeks', 'mean'),
            proj_update_latency        = ('latency',                   'mean'),
            proj_version_lag           = ('proj_version_lag',          'mean'),
            proj_criticality_scoreAVG  = ('criticality',               'mean'),
            proj_criticality_scoreSUM  = ('criticality',               'sum'),
            proj_alpha_best            = ('alpha/beta/rc',             'sum'),
            proj_alpha_used            = ('is_prerelease_new',         'sum'),
        )
        .reset_index()
    )

    for col in ['proj_mtbu','proj_update_latency','proj_version_lag','proj_criticality_scoreAVG', 'proj_criticality_scoreSUM']:
        agg[col] = agg[col].round(2)
    
    agg = agg.sort_values('commit_date').reset_index(drop=True)

    
    num_cols = ['proj_update_frequency', 'proj_mtbu', 'proj_update_latency', 'proj_version_lag', 'proj_criticality_scoreAVG']

    agg[num_cols] = agg[num_cols].where(agg[num_cols].notna(), 'N/A')
    agg['proj_criticality_scoreSUM'] = agg['proj_criticality_scoreSUM'].astype(object)
    mask = agg['proj_criticality_scoreAVG'] == 'N/A'
    agg.loc[mask, 'proj_criticality_scoreSUM'] = 'N/A'

    return agg


# ### 2.2 Process Commits (POM, Gradle, Py/JS)
# 
# Three parallel routines to walk each commit and process changed files by type:

# In[126]:


def process_commits_pom(repo, commits_with_changes):
    dependencies_over_time = {}
    dependencies_snapshot = {}

    for commit_hash, changed_files in tqdm(commits_with_changes.items(), desc="Processing commits"):
        dependencies_snapshot = process_commit_pom(repo, commit_hash, changed_files, dependencies_snapshot)
        dependencies_over_time[commit_hash] = dependencies_snapshot.copy()

    return dependencies_over_time


# In[127]:


def process_commits_gradle(repo, commits_with_changes):
    dependencies_over_time = {}
    dependencies_snapshot = {}
    all_dependencies = []
    properties_by_commit = {}

    for commit_hash, changed_files in tqdm(commits_with_changes.items(), desc="Processing commits"):
        dependencies_snapshot, all_dependencies = process_commit_gradle(
            repo, commit_hash, changed_files, dependencies_snapshot, all_dependencies, properties_by_commit
        )
        dependencies_over_time[commit_hash] = dependencies_snapshot.copy()

    return dependencies_over_time, all_dependencies



# In[128]:


def process_commits_py_js(repo_path, commits_with_files):
    for sha, files in tqdm(commits_with_files.items(), desc="Processing commits"):
        for file_path in files:
            if file_path.endswith("requirements.txt"):
                process_commit_file_content(repo_path, sha, file_path, "py")
            elif file_path.endswith("package.json"):
                process_commit_file_content(repo_path, sha, file_path, "js")
            elif file_path.endswith("setup.py"):
                process_commit_file_content(repo_path, sha, file_path, "py")
            elif file_path.endswith("pyproject.toml"):
                process_commit_file_content(repo_path, sha, file_path, "py")



# ### 2.3 Save Merged Data
# 
# A small helper to dump the combined dependency snapshot to disk:

# In[129]:


def save_merged_data(merged_data: Dict, output_file: str):
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)


# ### 2.4 Find Files
# 
# Functions to locate all POMs, Gradle files, `requirements.txt`, `package.json`, etc.:
# 

# In[130]:


def find_all_files(repo_path):
    found_files = []
    for root, dirs, file_list in os.walk(repo_path):
        for file in file_list:
            if file == "pom.xml" or file == "requirements.txt" or file == "package.json" or file == "setup.py" or file == "pyproject.toml":
                full_path = os.path.join(root, file)
                found_files.append(full_path)
    return found_files


# In[131]:


def find_all_gradle_files(repo_path):
    gradle_files = []
    patterns = [
        "build.gradle",
        "settings.gradle",
        "gradle.properties",
        "build.gradle.kts",
        "settings.gradle.kts",
        "*.gradle",
        "*.gradle.kts"
    ]

    for root, dirs, files in os.walk(repo_path):
        for file in files:
            for pattern in patterns:
                if fnmatch.fnmatch(file, pattern):
                    full_path = os.path.join(root, file)
                    gradle_files.append(full_path)
                    break
    return gradle_files


# ### 2.5 Commit History Helpers
# 
# Load commits and filter to those touching our target files:

# In[132]:


def get_all_relevant_commits(repo_path: str, file_paths: List[str]) -> Dict[str, List[str]]:
    repo = Repo(repo_path)
    commits_with_changes = {}

    for file_path in file_paths:
        rel_path = os.path.relpath(file_path, repo_path)

        # Get all commits that modified the file
        result = subprocess.run(
            ["git", "-C", repo_path, "log", "--follow", "--pretty=format:%H", "--name-only", "--", rel_path],
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.strip().split('\n')
        current_commit = None

        for line in lines:
            stripped_line = line.strip()
            
            if len(stripped_line) == 40 and all(c in '0123456789abcdef' for c in stripped_line):  
                current_commit = stripped_line
                if current_commit not in commits_with_changes:
                    commits_with_changes[current_commit] = set() 
            elif current_commit:
                modified_file = stripped_line
                if modified_file:  
                    commits_with_changes[current_commit].add(modified_file)

    commit_objects = []
    for commit_hash in commits_with_changes.keys():
        try:
            commit_obj = repo.commit(commit_hash)
            commit_objects.append((commit_obj, commit_hash))
        except BadName:
            print(f"Skipping invalid commit hash: {commit_hash}")  
    sorted_commits = sorted(commit_objects, key=lambda x: x[0].committed_date)
    
    sorted_commits_with_changes = {
        commit_hash: list(commits_with_changes[commit_hash]) for _, commit_hash in sorted_commits
        }
    
    global commits_with_date
    commits_with_date = {
        commit_hash: repo.commit(commit_hash).committed_datetime.strftime("%Y-%m-%d %H:%M:%S")
        for _, commit_hash in sorted_commits
    }

    return sorted_commits_with_changes


# ### 2.6 Parse POM XML
# 
# Extract `<dependency>` entries (groupId\:artifactId\:version):

# In[133]:


def parse_pom_xml(content: str, file_path: str, properties: Dict[str, str] = None) -> Dict[str, str]:
    if properties is None:
        properties = {}
    else:
        properties = properties.copy()

    dependencies = {}

    try:
        root = ET.fromstring(content)

        # Update namespace if needed
        if 'xmlns' in root.attrib:
            NAMESPACE['mvn'] = root.attrib['xmlns']

        # Load project properties
        for prop in root.findall(".//mvn:properties/*", NAMESPACE):
            if prop.tag and prop.text:
                prop_name = prop.tag.split('}')[-1]
                properties[prop_name] = prop.text.strip()
                properties_cache[prop_name] = properties[prop_name]
        
        # Load project and parent versions as fallback properties
        project_version = root.find(".//mvn:version", NAMESPACE)
        if project_version is not None and project_version.text:
            properties["project.version"] = project_version.text.strip()

        parent_version = root.find(".//mvn:parent/mvn:version", NAMESPACE)
        if parent_version is not None and parent_version.text:
            properties["parent.version"] = parent_version.text.strip()

        parent_info = {}
        relative_path_elem = root.find(".//mvn:parent/mvn:relativePath", NAMESPACE)
        if relative_path_elem is not None and relative_path_elem.text:
            parent_info = {"parent_pom_path": relative_path_elem.text.strip()}

        # Read dependencies
        for dependency in root.findall(".//mvn:dependency", NAMESPACE):
            group_id = dependency.find("mvn:groupId", NAMESPACE)
            artifact_id = dependency.find("mvn:artifactId", NAMESPACE)
            version = dependency.find("mvn:version", NAMESPACE)

            if group_id is not None and artifact_id is not None:
                dep_key = f"{group_id.text.strip()}:{artifact_id.text.strip()}"
                if "${" in dep_key and "}" in dep_key:
                    dep_key = (resolve_cached(dep_key))

                if version is not None and version.text:
                    version_text = version.text.strip()
                    if version_text.startswith("${") and version_text.endswith("}"):
                        prop_name = version_text[2:-1]
                        resolved_version = properties.get(prop_name, "UNRESOLVED")
                        if resolved_version == "UNRESOLVED":
                            resolved_version = resolve_from_parent(prop_name, file_path, parent_info)
                            if (resolved_version =="UNRESOLVED"):
                                resolved_version = properties_cache.get(prop_name, "UNRESOLVED")
                        if resolved_version.startswith("${") and resolved_version.endswith("}"):
                            resolved_version = resolve_cached(resolved_version)
                        dependencies[dep_key] = resolved_version
                    else:
                        dependencies[dep_key] = version_text
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
    return dependencies


# In[134]:


def resolve_cached(prop_name, visited=None):
    matches = re.findall(r"\$\{([^}]+)\}", prop_name)
    if not matches:
        return properties_cache.get(prop_name) 

    resolved = prop_name
    for match in matches:
        inner_value = resolve_cached(match)
        if inner_value is None:
            return match
        resolved = resolved.replace(f"${{{match}}}", inner_value)

    return resolved



# In[135]:


def resolve_from_parent(prop_name: str, file_path: str, properties: Dict[str, str]) -> str:
    parent_pom_path = properties.get("parent_pom_path")   
    if parent_pom_path is None:
        final_path = "pom.xml"
        parent_file = str(final_path).replace("\\", "/")
        if parent_file not in file_cache:
            return "UNRESOLVED"
        content = file_cache[parent_file]
        parent_prop_value = get_all_properties(content, prop_name, final_path, {})
        return parent_prop_value

    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
    parent_pom_path = Path(parent_pom_path) if not isinstance(parent_pom_path, Path) else parent_pom_path
    combined = file_path.parent / parent_pom_path

    stack = []
    for part in combined.parts:
        if part == "..":
            if stack and stack[-1] != "..":
                stack.pop()
            else:
                stack.append(part)
        elif part != ".":
            stack.append(part)
    final_path = Path(*stack)
    parent_file = str(final_path).replace("\\", "/")
    if parent_file not in file_cache:
        return "UNRESOLVED"
    content = file_cache[parent_file]
    parent_prop_value = get_all_properties(content, prop_name, final_path, {})
    return parent_prop_value


# In[136]:


def get_all_properties(content: str, target_prop: str, file_path: str, properties: Dict[str, str] = None):
    if properties is None:
        properties = {}
    else:
        properties = properties.copy()

    try:
        root = ET.fromstring(content)

        if 'xmlns' in root.attrib:
            NAMESPACE['mvn'] = root.attrib['xmlns']

        for prop in root.findall(".//mvn:properties/*", NAMESPACE):
            if prop.tag and prop.text:
                key = prop.tag.split('}')[-1]
                properties[key] = prop.text.strip()

        return properties.get(target_prop, "UNRESOLVED")
    except Exception as e:
        return "UNRESOLVED"    


# In[137]:


def process_commit_pom(repo, commit_hash, changed_files, dependencies_snapshot):
    for file_path in changed_files:
        if file_path.endswith("pom.xml"):
            content = load_file_at_commit_pom(repo, commit_hash, file_path)
            if content:
                new_dependencies = parse_pom_xml(content, file_path, properties)
                dependencies_snapshot[file_path] = new_dependencies
    return dependencies_snapshot


# In[138]:


def normalize_system(system: str) -> str:
    mapping = {
        "java": "maven",
        "javascript": "npm",
        "python": "pypi",
        "go": "go"
    }
    return mapping.get(system, system)


# In[139]:


def get_major_version(version_str: str):
    try:
        return Version(version_str).major
    except InvalidVersion:
        return None


# In[140]:


def fetch_package_data(package_name: str, system: str) -> dict:
    encoded = urllib.parse.quote(package_name, safe="")
    url = f"https://api.deps.dev/v3alpha/systems/{system}/packages/{encoded}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


# In[141]:


def get_cached_package_data(package_name: str, system: str) -> dict:
    LogFile = open("LOG_FILE.txt", "a")
    if package_name not in cached_versions[system]:
        try:
            data = fetch_package_data(package_name, system)
            try:
                versions = data.get("versions", [])
                if not versions:
                    raise RuntimeError(f"No versions found for {package_name}")

                chosen_version = versions[0].get("versionKey", {}).get("version")

                pkg_enc = urllib.parse.quote(package_name, safe="")
                ver_url = (
                    f"https://api.deps.dev/v3/systems/{system}"
                    f"/packages/{pkg_enc}/versions/{chosen_version}"
                )
                ver_resp = requests.get(ver_url)
                ver_resp.raise_for_status()
                version_data = ver_resp.json()

                rel = version_data.get("relatedProjects", [])
                if rel:
                    project_key = rel[0].get("projectKey", {}).get("id")
                else:
                    project_key = None

                if project_key and project_key.lower() != "null":
                    pk_enc = urllib.parse.quote(project_key, safe="")
                    ossf_url = f"https://api.deps.dev/v3/projects/{pk_enc}"
                    ossf_resp = requests.get(ossf_url)
                    if ossf_resp.ok:
                        data["ossf_score"] = ossf_resp.json().get("scorecard",{}).get("overallScore")
                        cached_versions[system][package_name] = data
                    else:
                        LogFile.write(f"[DEBUG] Failed to fetch OSSF score for {package_name} ({system}): {ossf_resp.status_code}\n")
                        data["ossf_score"] = None
                        cached_versions[system][package_name] = data
                else:
                    LogFile.write(f"[DEBUG] No related project found for {package_name} ({system})\n")
                    data["ossf_score"] = None
                    cached_versions[system][package_name] = data
                    
            except requests.RequestException as e:
                LogFile.write(f"[DEBUG] Failed to fetch version data for {package_name} ({system}): {e}\n")
                cached_versions[system][package_name] = data

        except (requests.RequestException, RuntimeError) as e:
            LogFile.write(f"[DEBUG] Failed to fetch data for {package_name} ({system}): {e}\n")
            cached_versions[system][package_name] = {}

    return cached_versions[system][package_name]



# In[142]:


def fetch_versions(group_id: str, artifact_id: str) -> list[str]:
    base_url = "https://repo1.maven.org/maven2"
    group_path = group_id.replace('.', '/')
    metadata_url = f"{base_url}/{group_path}/{artifact_id}/maven-metadata.xml"

    resp = requests.get(metadata_url)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    return [v.text for v in root.findall('versioning/versions/version')]


# In[143]:


def fetch_release_dates(group_id: str, artifact_id: str, versions: list[str]) -> dict[str, str]:
    base_url = "https://repo1.maven.org/maven2"
    group_path = group_id.replace('.', '/')
    release_dates: dict[str, str] = {}

    for version in versions:
        pom_url = f"{base_url}/{group_path}/{artifact_id}/{version}/{artifact_id}-{version}.pom"
        head = requests.head(pom_url)
        lm = head.headers.get('Last-Modified')
        if lm:
            dt = datetime.strptime(lm, '%a, %d %b %Y %H:%M:%S GMT')
            release_dates[version] = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            release_dates[version] = 'unknown'

    return release_dates


# In[144]:


def get_versions_with_dates_json(input_str: str) -> str:
    try:
        group, artifact = input_str.split(":")
        versions = fetch_versions(group, artifact)
        dates = fetch_release_dates(group, artifact, versions)

        items = {'versions': [
            {'versionKey': {'version': v}, 'publishedAt': dates[v]}
            for v in sorted(versions, key=lambda v: dates[v])
        ]}
        return json.dumps(items, indent=2)
    except Exception as e:
        return json.dumps({})


# In[145]:


def analyze_entry(entry: dict) -> dict:
    pkg        = entry["dependency"].split(" ")[0].split("[")[0]
    raw_new    = entry["new_version"].split(" ")[0].split("[")[0]
    entry["new_version"] = raw_new
    this_and_above  = raw_new.startswith("^") or raw_new.startswith("~=") or raw_new.startswith(">=") or raw_new.endswith("*") or raw_new == "latest-version-available"
    new_ver    = raw_new.lstrip("^").lstrip("~=").lstrip(">=").lstrip("~<").lstrip("<=").lstrip("<").lstrip("~").lstrip(">")
    
    commit_dt  = datetime.strptime(entry["date"], "%Y-%m-%d %H:%M:%S")
    sys_name   = normalize_system(entry["system"])
    is_npm = (sys_name == "npm")
    is_pypi = (sys_name == "pypi")


    data       = get_cached_package_data(pkg, sys_name)
    if data == {}:
        data = json.loads(get_versions_with_dates_json(pkg))
    if data == {}:
        entry["alpha/beta/rc"] = False
        entry["newest_version_at_commit"]= "N/A"
        entry["released_date"]= "N/A"
        entry["latest_at_commit_time"]= "N/A"
        entry["proj_version_lag"]= "N/A"
        entry["latency"]= "N/A"
        entry["criticality"]= "N/A"
        return entry

    all_vers   = data.get("versions", [])
    commit_vs  = []

    # Filter to releases published on or before commit date
    for v in all_vers:
        ver = v.get("versionKey", {}).get("version")
        pub = v.get("publishedAt")
        if not (ver and pub): continue
        try:
            pub_dt = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        if pub_dt <= commit_dt:
            try:
                commit_vs.append(Version(ver))
            except InvalidVersion:
                pass
        

    # Determine newest version at commit time
    entry["alpha/beta/rc"] = False

    if commit_vs and (is_npm or is_pypi):
        max_major = max(v.major for v in commit_vs)
        highest   = max(v for v in commit_vs if v.major == max_major)
        entry["newest_version_at_commit"] = str(highest)
        entry["newest_version_at_commit"] = entry["newest_version_at_commit"].replace("a", "-alpha.").replace("b", "-beta.").replace("rc", "-rc.")
        entry["alpha/beta/rc"] = highest.pre is not None
    else:
        entry["newest_version_at_commit"] = None

    entry["released_date"]         = None
    entry["latest_at_commit_time"] = None
    latest_pub_date = None

    target_major = get_major_version(new_ver) if is_npm else None

    try:
        baseline = Version(new_ver)
    except InvalidVersion:
        baseline = None

    if not this_and_above:
        for v in all_vers:
            ver = v.get("versionKey", {}).get("version")
            pub = v.get("publishedAt")
            if not (ver and pub): continue
            try:
                parsed = Version(ver)
            except InvalidVersion:
                parsed = None

            # npm: restrict to same major
            if (is_npm or is_pypi) and parsed and parsed.major != target_major:
                continue

            pub_dt = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ")
            if ver == new_ver:
                entry["released_date"] = pub_dt.isoformat()
            if pub_dt <= commit_dt and (latest_pub_date is None or pub_dt > latest_pub_date):
                latest_pub_date            = pub_dt
                entry["latest_at_commit_time"] = ver
    else:
        release_target = entry["newest_version_at_commit"]
        target_major = Version(release_target).major if release_target else None
        for v in all_vers:
            ver_str = v.get("versionKey", {}).get("version")
            pub = v.get("publishedAt")
            if not (ver_str and pub):
                continue
            try:
                pub_dt = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ")
                ver_obj = Version(ver_str)
            except (ValueError, InvalidVersion):
                continue
            # restrict to same major
            if target_major is not None and ver_obj.major != target_major:
                continue
            if ver_str == release_target:
                entry["released_date"] = pub_dt.isoformat()

    try:
        target = Version(new_ver)
    except InvalidVersion:
        target = None

    if entry["released_date"] is None and target is not None:
        for v in all_vers:
            ver_str = v.get("versionKey", {}).get("version")
            pub     = v.get("publishedAt")
            if not (ver_str and pub):
                continue

            try:
                ver_obj = Version(ver_str)
                pub_dt  = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, InvalidVersion):
                continue

            

            # compare by base_version so 1.2.3b1 == 1.2.3
            if ver_obj.release[:len(target.release)] == target.release:
                entry["released_date"] = pub_dt.isoformat()
                break
    rel_date_str = entry.get("released_date")
    released_date_dt = datetime.fromisoformat(rel_date_str) if rel_date_str else None    

    if this_and_above and baseline:
        same_maj = [v for v in commit_vs if v.major == baseline.major]
        if same_maj:
            baseline = max(same_maj)

    if baseline and commit_vs:
        entry["proj_version_lag"] = sum(1 for v in commit_vs if v > baseline)
    else:
        entry["proj_version_lag"] = -1
        for v in all_vers:
            version = v.get("versionKey").get("version")
            v_date_str =  v.get("publishedAt")
            if not v_date_str or not version:
                continue
            v_date = datetime.strptime(v_date_str, "%Y-%m-%dT%H:%M:%SZ")
            if released_date_dt and v_date >= released_date_dt:
                if v_date <= commit_dt:
                    entry["proj_version_lag"] += 1
        if entry["proj_version_lag"] == -1:
            entry["proj_version_lag"] = "N/A"
    

    commit_str = entry.get("date")
    commit_dt  = datetime.strptime(commit_str,  "%Y-%m-%d %H:%M:%S")

    release_str = entry.get("released_date")
    if release_str is None:
        entry["latency"] = "N/A"
        return entry
    

    release_dt = datetime.strptime(release_str,  "%Y-%m-%dT%H:%M:%S")
    if release_dt >= commit_dt:
        #entry["latency"] = "future: " + str((commit_dt - release_dt).days)
        #entry["proj_version_lag"] = "-1"
        entry["latency"] = "N/A"
        entry["proj_version_lag"] = "N/A"
    else:
        entry["latency"] = (commit_dt - release_dt).days

    try:
        parsed_new = Version(new_ver)
        entry["is_prerelease_new"] = parsed_new.pre is not None
    except InvalidVersion:
        entry["is_prerelease_new"] = False

    if entry["new_version"] == "latest-version-available":
        entry["latency"] = 0
        entry["proj_version_lag"] = 0

    entry["criticality"] = data.get("ossf_score", "N/A")

    return entry


# In[146]:


def process_json_file(input_path: str, output_path: str):
    """
    Read the input JSON, enrich each entry, and write to output.
    """
    with open(input_path, "r") as f:
        entries = json.load(f)

    enriched = []
    for i, e in enumerate(entries, 1):
        enriched.append(analyze_entry(e))

    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)


# ### 2.7 Load File at Commit
# 
# Three loaders—one per file‐type—to checkout a SHA and read content:

# In[147]:


def load_file_at_commit_pom(repo, commit_hash, file_path):
    try:
        commit = repo.commit(commit_hash)
        blob = commit.tree / file_path
        content = blob.data_stream.read().decode('utf-8')
        file_cache[file_path] = content
        return content
    except Exception as e:
        print(f"[DEBUG]Error loading file '{file_path}' at commit '{commit_hash}")    
        return None


# In[148]:


def load_file_at_commit_py_js(repo_path, commit_hash, file_path):
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "show", f"{commit_hash}:{file_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        try:
            LogFile.write(f"[DEBUG] Error loading file '{file_path}' at commit '{commit_hash}': {e.stderr}\n")
            LogFile.flush()
        except Exception as e:
            pass
        return None



# In[149]:


def load_file_at_commit_gradle(repo, commit_hash, file_path):
    try:
        commit = repo.commit(commit_hash)
        blob = commit.tree / file_path
        content = blob.data_stream.read().decode('utf-8')
        file_cache_python[file_path] = content
        return content
    except Exception as e:
        print(f"[DEBUG] Error loading file '{file_path}' at commit '{commit_hash}': {e}")
        return None
    


# ### 2.8 Python/JS Parsing Helpers
# 
# Split `requirements.txt`, `setup.py`, `pyproject.toml` or `package.json` into dependency dicts:

# In[150]:


def process_commit_file_content(repo_path, sha, file_path, type):
    content = load_file_at_commit_py_js(repo_path, sha, file_path)
        
    if content:
        parse_file_content(content, sha, file_path, type)


# In[151]:


def parse_pyproject_toml(content: str) -> dict:
    try:
        data = tomli.loads(content)
    except tomli.TOMLDecodeError:
        return {}

    deps = data.get("project", {}).get("dependencies", [])
    if not deps:
        poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        deps = [f"{pkg}{v if isinstance(v, str) else ''}" for pkg, v in poetry_deps.items() if pkg.lower() != "python"]

    result = {}
    for dep in deps:
        parts = re.split(r"([<>=!~^]+)", dep, maxsplit=1)
        name = parts[0].strip()
        version = ''.join(parts[1:]).strip() if len(parts) > 1 else "latest-version-available"
        result[name] = version

    return result



# In[152]:


def parse_setup_py(content: str) -> dict:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {}

    class SetupVisitor(ast.NodeVisitor):
        def __init__(self):
            self.install_requires = []

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "setup":
                for keyword in node.keywords:
                    if keyword.arg == "install_requires":
                        if isinstance(keyword.value, (ast.List, ast.Tuple)):
                            for elt in keyword.value.elts:
                                if isinstance(elt, ast.Str):
                                    self.install_requires.append(elt.s)
            self.generic_visit(node)

    visitor = SetupVisitor()
    visitor.visit(tree)

    result = {}
    for dep in visitor.install_requires:
        parts = re.split(r"([<>=!~^]+)", dep, maxsplit=1)
        name = parts[0].strip()
        version = ''.join(parts[1:]).strip() if len(parts) > 1 else "latest-version-available"
        result[name] = version

    return result


# In[153]:


def parse_file_content(content, sha, filename, type):
    dependencies = {}
    hash_pattern = re.compile(r"--hash=sha256:[a-fA-F0-9]{64}")

    if type == "py" and "requirements.txt" in filename:
        for line in content.splitlines():
            line = line.strip().split(";", 1)[0]

            if not line or line.startswith('#') or line.startswith("--"):  
                continue

            if hash_pattern.match(line):
                continue

            if '#' in line:
                line = line.split('#', 1)[0].strip()

            # Find the version specifier used
            version = "UNSPECIFIED"
            for specifier in VERSION_SPECIFIERS:
                if specifier in line:
                    package, version = line.split(specifier, 1)
                    package = package.strip()
                    version = version.strip()
                    dependencies[package] = version
                    break
            else:
                package = line
                dependencies[package] = "latest-version-available"

    elif type == "js" and "package.json" in filename:
        try:
            parsed_json = json.loads(content)
            if "dependencies" in parsed_json:
                dependencies.update(parsed_json["dependencies"])
            if "devDependencies" in parsed_json:
                dependencies.update(parsed_json["devDependencies"])
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file '{filename}': {e}")
    
    elif type == "py" and "setup.py" in filename:
        dependencies = parse_setup_py(content)
    elif type == "py" and "pyproject.toml" in filename:
        dependencies = parse_pyproject_toml(content)


    # Store the parsed dependencies for this commit
    dependency_changes[sha] = {
        "filename": filename,
        "dependencies": dependencies,
        "ecosystem": type
    }



# In[154]:


def save_results_python(output_file=f"{prefix}_python_and_javascript.json"):
    with open(output_file, "w") as f:
        json.dump(dependency_changes, f, indent=4)


# ### 2.9 Gradle Parsing Helpers
# 
# Extract Gradle `dependencies { … }` blocks, resolve property variables, group them, etc.:

# In[155]:


def is_gradle_related(file_path):
    return (
        file_path.endswith("build.gradle")
        or file_path.endswith("build.gradle.kts")
        or file_path.endswith("settings.gradle")
        or file_path.endswith("settings.gradle.kts")
        or file_path.endswith("gradle.properties")
        or file_path.endswith(".gradle")
        or file_path.endswith(".gradle.kts")
    )


# In[156]:


def parse_gradle_properties(content):
    props = {}
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            props[key.strip()] = value.strip()
    return props



# In[157]:


def extract_dependencies_block(content: str) -> str:
    """Returns only the content inside dependencies { ... } block(s)."""
    lines = content.splitlines()
    inside = False
    depth = 0
    collected = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("dependencies"):
            if "{" in stripped:
                inside = True
                depth = stripped.count("{") - stripped.count("}")
                continue

        if inside:
            if "{" in stripped:
                depth += stripped.count("{")
            if "}" in stripped:
                depth -= stripped.count("}")
            collected.append(line)
            if depth <= 0:
                inside = False

    return "\n".join(collected)



# In[158]:


def parse_gradle_dependencies(file_content: str, properties: Dict[str, str], commit_sha: str, file_path: str) -> List[Dict[str, Any]]:
    dependencies = []

    # extract only content inside dependencies block(s)
    block = extract_dependencies_block(file_content)

    # Patterns for Kotlin + Groovy
    patterns = [
        re.compile(r'(?P<config>\w+)\s*\(\s*["\'](?P<group>[^:\'"]+):(?P<artifact>[^:\'"]+):(?P<version>[^\'"]+)["\']\s*\)'),  # Kotlin DSL
        re.compile(r'(?P<config>\w+)\s+["\'](?P<group>[^:\'"]+):(?P<artifact>[^:\'"]+):(?P<version>[^\'"]+)["\']')  # Groovy DSL
    ]

    for pattern in patterns:
        for match in pattern.finditer(block):
            config = match.group("config")
            group = match.group("group")
            artifact = match.group("artifact")
            version = match.group("version")

            version_source = "literal"
            if version.startswith("$") or "${" in version:
                var_name = version.replace("$", "").replace("{", "").replace("}", "")
                if var_name in properties:
                    version = properties[var_name]

            if "$" not in group and "$" not in artifact and "$" not in version:
                dependencies.append({
                    "commit": commit_sha,
                    "file": file_path,
                    "group": group,
                    "artifact": artifact,
                    "version": version,
                })

    return dependencies



# In[159]:


def parse_local_kotlin_variables(content: str) -> Dict[str, str]:
    props = {}
    # Matches lines like: val log4jVersion = "2.20.0" or var version = "1.0"
    pattern = re.compile(r'(val|var)\s+(\w+)\s*=\s*["\']([^"\']+)["\']')
    for match in pattern.finditer(content):
        var_name = match.group(2)
        value = match.group(3)
        props[var_name] = value
    return props


# In[160]:


def process_commit_gradle(repo, commit_hash, changed_files, dependencies_snapshot, all_dependencies, properties_by_commit):
    for file_path in changed_files:
        if not is_gradle_related(file_path):
            continue

        content = load_file_at_commit_gradle(repo, commit_hash, file_path)
        if content:
            dependencies_snapshot[file_path] = content

            # Parse gradle.properties
            if file_path.endswith("gradle.properties"):
                props = parse_gradle_properties(content)
                properties_by_commit[commit_hash] = props

            # Parse dependencies from .gradle or .gradle.kts files
            elif file_path.endswith(".gradle") or file_path.endswith(".gradle.kts"):
                props = {}
                props.update(parse_local_kotlin_variables(content))

                if commit_hash not in properties_by_commit:
                    properties_by_commit[commit_hash] = {}
                properties_by_commit[commit_hash].update(props)

                resolved_props = properties_by_commit[commit_hash]
                parsed_deps = parse_gradle_dependencies(content, resolved_props, commit_hash, file_path)
                all_dependencies.extend(parsed_deps)

    return dependencies_snapshot, all_dependencies


# In[161]:


def group_gradle_dependencies(dependencies):
    grouped = defaultdict(lambda: {"gradle": defaultdict(dict)})

    for dep in dependencies:
        commit = dep["commit"]
        file = dep["file"]
        group_artifact = f"{dep['group']}:{dep['artifact']}"
        version = dep["version"]

        grouped[commit]["gradle"][file][group_artifact] = version

    return grouped


# In[162]:


def to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: to_dict(v) for k, v in d.items()}
    return d


# In[163]:


def save_results_gradle(data, output_file=f"{prefix}_dependencies_over_time.json"):
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


# ### 2.10 Merge & Clean Results
# 
# Utilities to deep‐merge the three language‐specific dicts and prune empty entries:

# In[164]:


def merge_dependencies(python_and_javascript: Dict, java: Dict, gradle: Dict) -> Dict:
    merged_data = {}

    # Combine all unique commit hashes from both datasets
    all_commits = set(python_and_javascript) | set(java) | set(gradle)

    for commit in all_commits:
        merged_data[commit] = {
            "java": {},
            "gradle": {},
            "python": {},
            "javascript": {}
        }

        # Add Java dependencies if present
        if commit in java:
            for file, content in java[commit].items():
                merged_data[commit]["java"][file] = content
        
        if commit in gradle:
            gradle_files = gradle[commit].get("gradle", {})  # Extract only the inner part
            for file, content in gradle_files.items():
                merged_data[commit]["gradle"][file] = content

        # Add Python and JavaScript dependencies if present
        if commit in python_and_javascript:
            content = python_and_javascript[commit]

            # Check if it's a Python or JavaScript file based on 'ecosystem'
            if content.get("ecosystem") == "py":
                merged_data[commit]["python"][content["filename"]] = content["dependencies"]
            elif content.get("ecosystem") == "js":
                merged_data[commit]["javascript"][content["filename"]] = content["dependencies"]


    return merged_data


# In[165]:


def remove_empty_objects(data):
    if isinstance(data, dict):
        return {
            k: remove_empty_objects(v)
            for k, v in data.items()
            if not (isinstance(v, dict) and not v)
        }
    elif isinstance(data, list):
        return [remove_empty_objects(item) for item in data]
    else:
        return data


# ## 3. Execution Workflow
# 
# This final section wires everything together:
# 
# ### 3.1 Initialize & Clone

# In[166]:


sep = "=" * 80
seq = "-" * 80
LogFile = open("LOG_FILE.txt", "a", encoding="utf-8") 
LogFile.write(sep + "\n")
LogFile.write(f"   Processing repository: {repo_name}\n")
LogFile.write(f"   Started at: {datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
LogFile.write(sep + "\n\n")
LogFile.flush()


# In[167]:


if not os.path.exists(destination_path) or not os.listdir(destination_path):
	Repo.clone_from(repo_url, destination_path)
else:
	print(f"Directory '{destination_path}' already exists and is not empty. Skipping clone.")


# In[168]:


repo = Repo(destination_path)


# ### 3.2 Discover Files
# 

# In[169]:


try:
    LogFile.write("[INFO] LOOKING FOR FILES")
    LogFile.flush()
except Exception as e:
    pass

file_paths = find_all_files(destination_path)
file_paths_gradle = find_all_gradle_files(destination_path)
file_paths += file_paths_gradle


try:
    LogFile.write(f"\n[DEBUG] Found {len(file_paths)} files \n")
    LogFile.flush()
except Exception as e:
    pass

# Display all found pom.xml paths
for path in file_paths:
    path = path.replace('\\', '/')
    try:
        LogFile.write(f"[DEBUG] Found file: {path}\n")
        LogFile.flush()
    except Exception as e:
        pass

try:
    LogFile.write(f"{seq}\n")
    LogFile.flush()
except Exception as e:
    pass


# ### 3.3 List Commits
# 

# In[170]:


try:
    LogFile.write("[INFO] LOOKING FOR COMMITS\n")
    LogFile.flush()
except Exception as e:
    pass

sorted_commit_hashes = get_all_relevant_commits(repo_path, file_paths)

try:
    LogFile.write(f"[DEBUG] Total number of commits: {len(sorted_commit_hashes)}\n")
    LogFile.flush()
except Exception as e:
    pass


for commit_hash, files in list(sorted_commit_hashes.items()):
    try:
        LogFile.write(f"[DEBUG] {commit_hash}: {files}\n")
        LogFile.flush()
    except Exception as e:
        pass


# ### 3.4 Parse Java-POM
# 

# In[171]:


try:
    LogFile.write(f"{seq}\n")
    LogFile.write(f"[INFO] PROCESSING JAVA-POM COMMITS \n")
    LogFile.flush()
except Exception as e:
    pass

dependencies_over_time = process_commits_pom(repo, sorted_commit_hashes)
try:
    LogFile.write(f"[DEBUG] Processed {len(dependencies_over_time)} commits\n")
except Exception as e:
    pass

with open(f"{prefix}_java_pom.json", "w") as f:
    json.dump(dependencies_over_time, f, indent=4)

try:
    LogFile.write(f"[DEBUG] Output temporarily stored in {prefix}_java_pom.json\n")
    LogFile.write(f"{seq}\n")
    LogFile.flush()
except Exception as e:
    pass


# ### 3.5 Parse Python & JS
# 

# In[172]:


file_cache_python = {}
unique_entries = set()
dependency_changes = defaultdict(list)

# Supported version specifiers in requirements.txt
VERSION_SPECIFIERS = [
    '==', '>=', '<=', '~=', '!=', '>', '<' '^'
]


# In[173]:


try:
    LogFile.write(f"[INFO] PROCESSING PYTHON AND JAVASCRIPT COMMITS\n")
    LogFile.flush()
except Exception as e:
    pass
process_commits_py_js(repo_path, sorted_commit_hashes)
save_results_python()
try:
    LogFile.write(f"[DEBUG] Processed {len(dependency_changes)} commits\n")
    LogFile.write(f"[DEBUG] Output temporarily stored in {prefix}_python_and_javascript.json\n")
    LogFile.write(f"{seq}\n")
    LogFile.flush()
except Exception as e:
    pass


# ### 3.6 Parse Java-Gradle
# 

# In[174]:


try:
    LogFile.write(f"[INFO] PROCESSING JAVA-GRADLE COMMITS\n")
    LogFile.flush()
except Exception as e:
    pass

dependencies_over_time, all_dependencies = process_commits_gradle(repo, sorted_commit_hashes)

cleaned_dependencies = [
    dep for dep in all_dependencies
    if "$" not in dep["group"]
    and "$" not in dep["artifact"]
    and "$" not in dep["version"]
]

try:
    LogFile.write(f"[DEBUG] Processed {len(dependencies_over_time)} commits\n")
    LogFile.flush()
except Exception as e:
    pass


# In[175]:


grouped_dependencies = group_gradle_dependencies(cleaned_dependencies)


# In[176]:


save_results_gradle(to_dict(grouped_dependencies), f"{prefix}_java_gradle.json")

try:
    LogFile.write(f"[DEBUG] Output temporarily stored in {prefix}_java_gradle.json\n")
    LogFile.write(f"{seq}\n")
    LogFile.flush()
except Exception as e:
    pass


# ### 3.7 Merge & Save
# 

# In[177]:


with open(f'{prefix}_python_and_javascript.json', 'r') as f:
    python_and_javascript = json.load(f)

with open(f'{prefix}_java_pom.json', 'r') as f:
    java = json.load(f)

with open(f'{prefix}_java_gradle.json', 'r') as f:
    gradle = json.load(f)

try:
    LogFile.write(f"[INFO] MERGING DEPENDENCIES\n")
    LogFile.flush()
except Exception as e: 
    pass

merged_data = merge_dependencies(python_and_javascript, java, gradle)
new_merged_data = remove_empty_objects(merged_data)
save_merged_data(new_merged_data, f'{prefix}_merged_dependencies.json')


# 
# ## 4. Results & Calculated Metrics
# 
# 

# In[178]:


try:
    LogFile.write(f"[DEBUG] Merged data saved to {prefix}_merged_dependencies.json\n")
    LogFile.flush()
except Exception as e:
    pass

with open(f"{prefix}_merged_dependencies.json", "r") as f:
    data = json.load(f)

sorted_commits = sorted(data.keys(), key=lambda c: commits_with_date.get(c))
latest_versions = {}
last_file_state = {}
changes = []

for commit in sorted_commits:
    date = commits_with_date.get(commit)
    commit_data = data[commit]

    for system, files in commit_data.items():
        for filename, curr_deps in files.items():
            file_key = (system, filename)
            prev_deps = last_file_state.get(file_key, {})

            for dep, new_version in curr_deps.items():
                old_version = latest_versions.get(dep)
                if dep not in prev_deps:
                    changes.append({
                        "commit": commit,
                        "date": date,
                        "filename": filename,
                        "system": system,
                        "dependency": dep,
                        "change_type": "added",
                        "old_version": None,
                        "new_version": new_version
                    })
                elif prev_deps[dep] != new_version:
                    changes.append({
                        "commit": commit,
                        "date": date,
                        "filename": filename,
                        "system": system,
                        "dependency": dep,
                        "change_type": "updated",
                        "old_version": prev_deps[dep].split(" ")[0],
                        "new_version": new_version.split(" ")[0]
                    })

                latest_versions[dep] = new_version

            last_file_state[file_key] = copy.deepcopy(curr_deps)

with open(f"{prefix}_dependency_changes_with_removed.json", "w") as f:
    json.dump(changes, f, indent=2)

try:
    LogFile.write(f"[DEBUG] All detected changes saved to {prefix}_dependency_changes_with_removed.json\n")
    LogFile.flush()
except Exception as e:
    pass


# In[179]:


try:
    LogFile.write(f"{seq}\n")
    LogFile.write(f"[INFO] ENRICHING DEPENDENCY CHANGES\n")
    LogFile.flush()
except Exception as e:
    pass

process_json_file(
        f"{prefix}_dependency_changes_with_removed.json",
        f"{prefix}_dependency_changes_enriched.json"
    )


# In[180]:


try:
    LogFile.write(f"{seq}\n")
    LogFile.write(f"[INFO] CALCULATING METRICS\n")
    LogFile.flush()
except Exception as e:
    pass
result = compute_per_commit_file_metrics(f"{prefix}_dependency_changes_enriched.json", weeks=13)

result.to_csv(f'{prefix}_metrics_output.csv',index=False, sep=',', encoding='utf-8')

try:
    LogFile.write(f"[DEBUG] Metrics saved to {prefix}_metrics_output.csv\n")
    LogFile.write(f"{seq}\n")
    LogFile.flush()
except Exception as e:
    pass


# In[181]:


try:
    LogFile.write(f"[INFO] COLLECTING DEPENDENCIES\n")
    LogFile.flush()
except Exception as e:
    pass


def compute_per_commit_file_metrics(input_path: str, weeks: int = 13) -> pd.DataFrame:    
    json_path  = f'{prefix}_dependency_changes_enriched.json'
    output_csv = f'{prefix}_commits_dependencies.csv'
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if df.empty:
            combined = pd.DataFrame(columns=['commit_hash', 'commit_date', 'commit_file', 'dependencies'])
            combined.to_csv(output_csv, index=False)
        detailed = (
            df
            .groupby(['commit', 'date', 'filename'], as_index=False)
            ['dependency']
            .agg(lambda deps: ';'.join(sorted(set(deps))))
            .rename(columns={
                'commit': 'commit_hash',
                'date': 'commit_date',
                'filename': 'commit_file',
                'dependency': 'dependencies'
            })
        )
        all_deps = sorted(df['dependency'].dropna().unique())
        summary = pd.DataFrame([{
            'commit_hash': 'PROJECT_SUMMARY',
            'commit_date': '',
            'commit_file': '',
            'dependencies': ';'.join(all_deps)
        }])

        combined = pd.concat([detailed, summary], ignore_index=True)
        combined['commit_date'] = pd.to_datetime(combined['commit_date'], errors='coerce')
        combined = combined.sort_values('commit_date', na_position='last')
        combined['commit_date'] = combined['commit_date'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
        combined.to_csv(output_csv, index=False)
    except Exception as e:
        pass

compute_per_commit_file_metrics(f"{prefix}_dependency_changes_enriched.json", weeks=13)

try:
    LogFile.write(f"[DEBUG] Metrics saved to {prefix}_commits_dependencies.csv\n")
    LogFile.write(f"{seq}\n\n\n\n")
    LogFile.flush()
    LogFile.close()
except Exception as e:
    pass



# In[182]:


p = Path('.')
for file in p.glob(f"{prefix}_*.json"):
    file.unlink()
    print(f"Deleted {file}")


# In[ ]:




