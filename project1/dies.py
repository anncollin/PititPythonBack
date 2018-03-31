"""
    Contains all the die classes
"""


class Die:
    def __init__(self, max_adv=1, trap_prob=1.0, d_type=-1, name=""):
        self.adv = max_adv
        self.trap_prob = trap_prob
        self.type = d_type
        self.name = name+" die"

    def __hash__(self):
        return self.type

    def __str__(self):
        return self.name


class SecurityDie(Die):
    def __init__(self):
        super().__init__(1, 0., 1, "security")


class NormalDie(Die):
    def __init__(self):
        super().__init__(2, 0.5, 2, "normal")


class RiskyDie(Die):
    def __init__(self):
        super().__init__(3, 1.0, 3, "risky")


def get_all_dice():
    return [SecurityDie(), NormalDie(), RiskyDie()]

