def adder(tall1, tall2):
    tall3 = tall1 + tall2
    return tall3
def kall_opp_adder_tre_ganger():
    for i in range(2):
        tall1 = input('skrive første tall: ')
        tall2 = input('skrive andre tall: ')
        print('\n',tall1, ' + ', tall2, ' = ', adder(int(tall1), int(tall2)))


def forekomst(tekst, bokstav):
    antall_bokstav = tekst.count(bokstav)
    return antall_bokstav

def main():
    tekst = input('Skriv ett ord: ')
    bokstav = input('Skriv en bokstav: ')
    antall_bokstav = forekomst(tekst, bokstav)
    if tekst.find(bokstav) != -1:
        print('\n', bokstav + ' forekommer ' + str(antall_bokstav) + ' ganger i ' + tekst)
    else:
        print('\n', bokstav + ' er ikke i ' + tekst)

main()