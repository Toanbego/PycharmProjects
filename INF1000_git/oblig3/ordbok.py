varer = {
    "melk - kr": 14.90,
    "brød - kr": 24.90,
    "yoghurt - kr" : 12.90,
    "pizza - kr" : 39.90
}

for i in range(2):
    ny_vare = input('vare: ')
    pris = input('pris: ')
    pris = float(pris)
    varer[ny_vare] = pris

print(varer)