import sys
import json
import urllib.request
import urllib.error
import argparse

def fetch_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'unsloth-buddy/1.0'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def fetch_text(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'unsloth-buddy/1.0'})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="Search and fetch DESIGN.md templates from getdesign.md")
    parser.add_argument("keyword", help="Keyword to search for a brand design")
    parser.add_argument("--fetch", action="store_true", help="Download the matching DESIGN.md content")
    args = parser.parse_args()
    
    manifest_url = "https://unpkg.com/getdesign@latest/templates/manifest.json"
    try:
        manifest = fetch_json(manifest_url)
    except Exception as e:
        print(f"Error fetching manifest: {e}")
        sys.exit(1)

    keyword = args.keyword.lower()
    matches = [m for m in manifest if keyword in m['brand'].lower() or keyword in m.get('description', '').lower()]
    
    if not matches:
        print(f"No match found for '{args.keyword}'.")
        print("\nAvailable brands:")
        print(", ".join(m['brand'] for m in manifest))
        sys.exit(1)
        
    match = matches[0]
    print(f"Matched brand: {match['brand']}")
    print(f"Description: {match['description']}")
    
    if args.fetch:
        print(f"\n--- DESIGN.md for {match['brand']} ---\n")
        design_url = f"https://unpkg.com/getdesign@latest/templates/{match['file']}"
        try:
            design_md = fetch_text(design_url)
            print(design_md)
        except Exception as e:
            print(f"Error fetching template: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
