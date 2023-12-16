morphology = [
    '???',
    '[SEP]',
    ('V', [
        ('Abs/Erg', [
            ('Pl', [
                ('1P', [
                    'A1P',
                    'E1P',
                ]),
                ('2P', [
                    'A2P',
                    'E2P',
                ]),
                ('3P', [
                    'E3P',
                ])
            ]),
            ('Si', [
                ('1S', [
                    'A1S',
                    'E1S',
                ]),
                ('2S', [
                    'A2S',
                    'E2S',
                ]),
                ('3S', [
                    'E3S',
                ])
            ]),
            ('Any', [
                ('1', [
                    'E1',
                ]),
                ('2', [
                    'E2',
                ]),
                ('3', [
                    'E3',
                ])
            ])
        ]),
        ('Transitivity', [
            'AFE',
            'ITR',
            'TRN',
        ]),
        ('Voice', [
            'AP',
            'APLI',
            'CAU',
            'MOV',
            'PAS',
            'REC',
            'RFX',
        ]),
        ('TAM', [
            ('Aspect', [
                'COM',
                'INC',
                'PRG',
            ]),
            ('Mood', [
                'COND',
                'IMP',
                'INT',
            ]),
            'TAM',
        ]),
        ('Deriv', [
            'DIR',
            'PP',
            'SC',
            'SV',
        ]),
        ('Stem', [
            'EXS',
            'POS',
            'VI',
            'VT',
        ])
    ]),
    ('S', [
        ('Stem', [
            'ADJ',
            'NOM',
            'PRON',
            'S',
            'SAB',
            'TOP',
            'VOC',
        ]),
        ('Modifier', [
            'CLAS',
            'DIM',
            'GNT',
            'NUM',
        ]),
        ('Case', [
            'AGT',
            'INS',
        ]),
        'PL',
    ]),
    'ADV',
    ('PART', [
        'AFI',
        'NEG',
        'PART',
        'SREL',
    ]),
    ('DET', [
        'ART',
        'DEM',

    ]),
    'CONJ',
    ('DERIV', [
        'ENF',
        'ITS',
        'MED',
    ]),
    'PREP',
]

simplified_morphology = [
    '???',
    '[SEP]',
    ('Abs', [
        'A1P',
        'A2P',
        'A1S',
        'A2S'
    ]),
    ('Erg', [
        'E1',
        'E2',
        'E3',
        'E1P',
        'E2P',
        'E3P',
        'E1S',
        'E2S',
        'E3S',
    ]),
    ('TAM', [
        'COM',
        'INC',
        'PRG',
        'COND',
        'IMP',
        'INT',
        'TAM',
    ]),
    ('VOICE', [
        'AFE',
        'ITR',
        'TRN',
        'AP',
        'APLI',
        'CAU',
        'MOV',
        'PAS',
        'REC',
        'RFX',
    ]),
    'DIR',
    'PP',
    'SC',
    'SV',
    ('Stem', [
        'EXS',
        'POS',
        'VI',
        'VT',
        'ADJ',
        'NOM',
        'PRON',
        'S',
        'SAB',
        'TOP',
        'VOC',
    ]),
    'CLAS',
    'DIM',
    'GNT',
    'NUM',
    'AGT',
    'INS',
    'PL',
    'ADV',
    'AFI',
    'NEG',
    'PART',
    'SREL',
    'ART',
    'DEM',
    'CONJ',
    'ENF',
    'ITS',
    'MED',
    'PREP',
]
