morphology = [
    '???',
    '[SEP]',
    ('V', [
        ('Abs/Erg', [
            ('Abs', [
                ('Pl', [
                    'A1P',
                    'A2P',
                ]),
                ('Si', [
                    'A1S',
                    'A2S',
                ])
            ]),
            ('Erg', [
                ('Any', [
                    'E1',
                    'E2',
                    'E3',
                ]),
                ('Pl', [
                    'E1P',
                    'E2P',
                    'E3P',
                ]),
                ('Si', [
                    'E1S',
                    'E2S',
                    'E3S',
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
