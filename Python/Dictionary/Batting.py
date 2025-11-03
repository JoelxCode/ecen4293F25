import csv
from collections import defaultdict

def read_batting_data(filename):
    """Read batting data and return list of dictionaries"""
    batting_data = []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                year = int(row['yearID'])
                at_bats = int(row['AB'])
                hits = int(row['H'])
                player_id = row['playerID']
                
                # Filter: year >= 1939 and AB >= 500
                if year >= 1939 and at_bats >= 500:
                    batting_avg = hits / at_bats
                    batting_data.append({
                        'playerID': player_id,
                        'year': year,
                        'AB': at_bats,
                        'H': hits,
                        'avg': batting_avg
                    })
            except (ValueError, KeyError, ZeroDivisionError):
                continue
    
    return batting_data

def read_people_data(filename):
    """Read people data and return dictionary mapping playerID to name"""
    people = {}
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                player_id = row['playerID']
                first_name = row.get('nameFirst', '')
                last_name = row.get('nameLast', '')
                full_name = f"{first_name} {last_name}".strip()
                people[player_id] = full_name
            except KeyError:
                continue
    
    return people

def find_top_25_averages():
    """Find the top 25 batting averages since 1939 with AB >= 500"""
    
    # Read the data
    print("Reading Batting.csv...")
    batting_data = read_batting_data('Batting.csv')
    
    print("Reading People.csv...")
    people = read_people_data('People.csv')
    
    # Sort by batting average (descending)
    batting_data.sort(key=lambda x: x['avg'], reverse=True)
    
    # Get top 25
    top_25 = batting_data[:25]
    
    print(f"\nTop 25 Batting Averages Since 1939 (minimum 500 AB):")
    print("=" * 80)
    print(f"{'Rank':<4} {'Player':<25} {'Year':<6} {'AB':<5} {'H':<5} {'AVG':<8}")
    print("-" * 80)
    
    for i, record in enumerate(top_25, 1):
        player_name = people.get(record['playerID'], f"Unknown ({record['playerID']})")
        print(f"{i:<4} {player_name:<25} {record['year']:<6} {record['AB']:<5} "
              f"{record['H']:<5} {record['avg']:.6f}")
    
    # Check for Manny Ramirez in 2008
    print("\n" + "=" * 80)
    print("Checking Manny Ramirez in 2008:")
    
    manny_2008 = None
    for record in batting_data:
        if record['playerID'] == 'ramirma02' and record['year'] == 2008:
            manny_2008 = record
            break
    
    if manny_2008:
        player_name = people.get('ramirma02', 'Manny Ramirez')
        print(f"Found: {player_name}")
        print(f"Year: {manny_2008['year']}")
        print(f"At Bats: {manny_2008['AB']}")
        print(f"Hits: {manny_2008['H']}")
        print(f"Batting Average: {manny_2008['avg']:.8f}")
        
        if manny_2008['AB'] >= 500:
            # Find rank in our filtered data
            rank = next((i for i, r in enumerate(batting_data, 1) 
                        if r['playerID'] == 'ramirma02' and r['year'] == 2008), None)
            print(f"Rank among qualified seasons: {rank}")
            
            if rank and rank <= 25:
                print("✓ This appears in the top 25!")
            else:
                print("✗ This does not appear in the top 25.")
        else:
            print("✗ Does not meet minimum 500 AB requirement.")
    else:
        print("❌ Manny Ramirez not found in 2008 data or didn't meet criteria.")
    
    return top_25

if __name__ == "__main__":
    try:
        top_25 = find_top_25_averages()
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file - {e}")
        print("Make sure Batting.csv and People.csv are in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")