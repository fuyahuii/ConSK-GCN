class Sample:

    def __init__(self, vid, speaker, label, text, audio, visual, sentence):
        self.vid = vid # dialogue_id
        self.speaker = speaker
        self.label = label
        self.text = text
        self.audio = audio
        self.visual = visual
        self.sentence = sentence # max_len=110
