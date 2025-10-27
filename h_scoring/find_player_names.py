"""
Helper script to find player names in NBA API.
Useful for identifying name mismatches.
"""

from nba_api.stats.static import players

def find_players_by_partial_name(search_term):
    """Find players whose name contains the search term."""
    player_list = players.get_players()

    search_lower = search_term.lower()
    matches = [
        p for p in player_list
        if search_lower in p['full_name'].lower()
    ]

    return matches


if __name__ == "__main__":
    # Search for the problematic names
    search_names = [
        "jimmy butler",
        "nicolas claxton",
        "nic claxton",
        "alexandre sarr",
        "carlton carrington",
        "cooper flagg",
        "ace bailey",
        "dylan harper",
        "kon knueppel",
        "vj edgecombe"
    ]

    print("=" * 70)
    print("SEARCHING FOR PLAYERS IN NBA API")
    print("=" * 70)

    for search_name in search_names:
        print(f"\nSearching for: {search_name}")
        matches = find_players_by_partial_name(search_name)

        if matches:
            print(f"  Found {len(matches)} match(es):")
            for match in matches:
                status = "ACTIVE" if match['is_active'] else "inactive"
                print(f"    - {match['full_name']} (ID: {match['id']}, {status})")
        else:
            print(f"  ❌ No matches found (likely not in NBA API yet)")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
