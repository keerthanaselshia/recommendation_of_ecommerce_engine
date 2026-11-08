import urllib.request, json

tests = [
    ('GET /health',                                   'http://127.0.0.1:8000/health'),
    ('GET /compare?ids=P0001,P0002',                  'http://127.0.0.1:8000/compare?ids=P0001,P0002'),
    ('GET /compare?ids=P0001,P0002,P0010,P0050',      'http://127.0.0.1:8000/compare?ids=P0001,P0002,P0010,P0050'),
    ('GET /dashboard/compare.html',                   'http://127.0.0.1:8000/dashboard/compare.html'),
    ('GET /dashboard/index.html',                     'http://127.0.0.1:8000/dashboard/index.html'),
]

for label, url in tests:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = r.read()
            if 'compare?ids' in url:
                j = json.loads(data)
                n_products = len(j['products'])
                n_listings = sum(len(p['listings']) for p in j['products'])
                best       = j['best_overall']['badge']
                best_score = max(
                    l['score']
                    for p in j['products'] for l in p['listings']
                )
                print(f'  [OK] {label}')
                print(f'       {n_products} products  {n_listings} listings  best="{best}"  top_score={best_score}')
                # Print score breakdown for the best listing
                for p in j['products']:
                    for l in p['listings']:
                        if l['listing_id'] == j['best_overall']['listing_id']:
                            bd = l['score_breakdown']
                            print(f'       breakdown: rating={bd["rating_score"]} price={bd["price_score"]} review={bd["review_score"]} offer={bd["offer_score"]}')
            else:
                print(f'  [OK] {label}  ({len(data):,} bytes)')
    except Exception as e:
        print(f'  [ERR] {label}')
        print(f'         {e}')

print('\nAll done.')
