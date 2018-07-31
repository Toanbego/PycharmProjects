steder = []
klesplagg = []
avreisedatoer = []
def reisepl():
    print('Skriv inn 5 reisemål: ')
    for i in range(5):
        sted = input()
        steder.append(sted)

    print('Skriv inn 5 klesplagg: ')
    for i in range(5):
        plagg = input()
        klesplagg.append(plagg)

    print('Skriv inn 5 avreisedatoer: ')
    for i in range(5):
        datoer = input()
        avreisedatoer.append(datoer)

    reiseplan = [steder, klesplagg, avreisedatoer]
    print(len(reiseplan))
    for elements in reiseplan:
        print(elements)

reise = [['oslo', 'budapest', 'mallorca', 'valencia', 'bergen'],
['genser', 'bukse', 'sokker', 'undertoy', 'skjorte'],
['0908', '0807', '0604', '234', '23423']]
print('skriv inn indeks1 og indeks2: ')
i1 = input()
i2 = input()

try:
    i1, i2 = int(i1), int(i2)
    try:
        print(reise[i1][i2])
    except IndexError:
        print('ugyldig input')

except ValueError:
    print('Det ble ikke skrevet inn ett tall')

