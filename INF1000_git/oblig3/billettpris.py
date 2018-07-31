
def sjekkAlder (alder):
    billettpris = 0


    if int(alder) <=17:
        billettpris = 30
    elif int(alder) >= 63:
        billettpris = 35
    else:
        billettpris = 50
    return billettpris

for i in range(4):

    alder = input('Hei, hvor gammel er du? ')
    if int(alder) <= 17:
        print('Barnebillett: ', str(sjekkAlder(alder))+'.')
    elif int(alder) >= 63:
        print('Seniorbillett: ', str(sjekkAlder(alder))+'.')
    else:
        print('Ordinærbillett: ', str(sjekkAlder(alder))+'.')
