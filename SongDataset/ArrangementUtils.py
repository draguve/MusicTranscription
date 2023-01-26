import torch

arrangementsToConvert = ["lead", "lead2", "lead3", "rhythm", "rhythm2", "rhythm3"]
arrangementIndex = {x: index - (len(arrangementsToConvert) / 2) for index, x in enumerate(arrangementsToConvert)}


def getTuning(offsets: list[int]):
    assert len(offsets) == 6
    return torch.Tensor(offsets)


def getArrangementTensor(tuning: torch.tensor, arrangement: str, capo: float):
    torch.cat((tuning, torch.Tensor(arrangementIndex[arrangement]),
               torch.Tensor([float(capo)])))


E_Standard = getTuning([0] * 6)
DSharp_Standard = getTuning([-1] * 6)
D_Standard = getTuning([-2] * 6)
CSharp_Standard = getTuning([-3] * 6)

if __name__ == '__main__':
    print(getTuning([0] * 6))
