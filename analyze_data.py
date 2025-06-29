#!/usr/bin/env python3
"""
Simple data analysis script for GenerativeLife simulation data.
Analyzes agent behavior patterns from collected session data.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict


def analyze_session_data(session_file):
    """Analyze a single session file."""
    print(f"\n📊 Analyzing: {session_file}")
    print("=" * 50)

    data = []
    try:
        with open(session_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    if not data:
        print("❌ No data found in file")
        return

    # Basic statistics
    print(f"📈 Total decisions: {len(data)}")

    # Decision analysis
    decisions = [entry['decision'] for entry in data]
    decision_counts = Counter(decisions)

    print(f"\n🎯 Decision Distribution:")
    for decision, count in decision_counts.most_common():
        percentage = (count / len(decisions)) * 100
        print(f"  {decision}: {count} ({percentage:.1f}%)")

    # Agent analysis
    agents = defaultdict(list)
    for entry in data:
        agents[entry['agent_id']].append(entry)

    print(f"\n👥 Agent Analysis:")
    print(f"  Total agents: {len(agents)}")

    for agent_id, agent_data in agents.items():
        genders = [entry['agent_info']['gender'] for entry in agent_data]
        personalities = [entry['agent_info']['personality']
                         for entry in agent_data]

        print(f"  {agent_id}: {len(agent_data)} decisions")
        print(f"    Gender: {genders[0]}")
        print(f"    Personality: {personalities[0]}")

        # Energy progression (robust to missing energy)
        energies = [entry['agent_info'].get(
            'energy', None) for entry in agent_data if 'energy' in entry['agent_info']]
        if energies:
            print(
                f"    Energy: {energies[0]} → {energies[-1]} (Δ{energies[-1] - energies[0]})")
        else:
            print("    Energy: (no energy data in agent_info)")

    # Compound action analysis
    compound_actions = [d for d in decisions if ',' in d]
    if compound_actions:
        print(f"\n⚡ Compound Actions:")
        print(
            f"  Total: {len(compound_actions)} ({len(compound_actions)/len(decisions)*100:.1f}%)")
        compound_counts = Counter(compound_actions)
        for action, count in compound_counts.most_common():
            print(f"    {action}: {count}")

    # Success rate analysis
    successes = [entry['outcome']['success']
                 for entry in data if 'outcome' in entry]
    if successes:
        success_rate = sum(successes) / len(successes) * 100
        print(
            f"\n✅ Success Rate: {success_rate:.1f}% ({sum(successes)}/{len(successes)})")


def main():
    """Main analysis function."""
    print("🔍 GenerativeLife Data Analysis Tool")
    print("=" * 50)

    # Find data files
    data_dir = Path("data/sessions")
    if not data_dir.exists():
        print("❌ No data/sessions directory found")
        return

    session_files = list(data_dir.glob("*.jsonl"))
    if not session_files:
        print("❌ No session data files found")
        return

    print(f"📁 Found {len(session_files)} session files")

    # Analyze each session file
    for session_file in sorted(session_files):
        analyze_session_data(session_file)

    print("\n" + "=" * 50)
    print("🎉 Analysis complete!")
    print("\nTo run a new simulation and collect more data:")
    print("  python ai_arena.py --agents 5 --turns 50")


if __name__ == "__main__":
    main()
