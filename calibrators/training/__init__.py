import torch
from torch import nn
from .loss import EntropyRegularAttn, GraduatedLabelSmoothingAttn, SequenceSmoothLoss_v10, CrossEntropyLoss, SequenceSmoothLoss_v11, LabelSmoothingLoss, LogitMarginL1, ClassficationAndMDCA, FocalLoss, SequenceSmoothLoss_v12
from .loss_ctc import CTC_Loss, SequenceSmoothLossCTC_PAM_wo_adapt, SequenceSmoothLossCTC_PAMI, CTCClassficationAndMDCA, CTCEntropyRegular, CTCGraduatedLabelSmoothing, CTCLogitMarginL1, SequenceSmoothLossCtc_v10, CTCLabelSmoothLoss
LOSS_FUNC={
    'CE':CrossEntropyLoss,
    'LS':LabelSmoothingLoss,
    'SeqLSv1_0':SequenceSmoothLoss_v10,
    'SeqLSv1_1':SequenceSmoothLoss_v11,
    'SeqLSv1_2':SequenceSmoothLoss_v12,
    'MBLS':LogitMarginL1,
    'MDCA':ClassficationAndMDCA,
    'FL':FocalLoss,
    'ER':EntropyRegularAttn,
    'GLS':GraduatedLabelSmoothingAttn
}

LOSS_FUNC_CTC={
    'CTC':CTC_Loss,
    'CTCLS':CTCLabelSmoothLoss,
    'SeqLSv1_0_ctc':SequenceSmoothLossCtc_v10, 
    'SeqLS_ctc_pami':SequenceSmoothLossCTC_PAMI, 
    'SeqLS_ctc_pami_wo_adapt':SequenceSmoothLossCTC_PAM_wo_adapt, 
    'MDCA':CTCClassficationAndMDCA,
    'MBLS':CTCLogitMarginL1,
    'ER':CTCEntropyRegular,
    'GLS':CTCGraduatedLabelSmoothing
}