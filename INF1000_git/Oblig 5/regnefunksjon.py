def addisjon(heltall1, heltall2):
    sum = heltall1 + heltall2
    return sum
def substraksjon(heltall1, heltall2):
    sum = heltall1 - heltall2
    return sum
def divisjon(heltall1, heltall2):
    sum = heltall1 / heltall2
    return sum

def tommer_til_cm(tommer):
    assert tommer > 0
    return tommer*2.54

tommer_til_cm(1)

def check_functions():
    assert addisjon(3, 5 ) == 8
    assert addisjon(-3, -5) == -8
    assert addisjon(3, -5) == -2

    assert substraksjon(3, 5 ) == -2
    assert substraksjon(-3, -5) == 2
    assert substraksjon(3, -5) == 8

    assert divisjon(4, 2 ) == 2
    assert divisjon(-15, -5) == 3
    assert divisjon(15, -5) == -3