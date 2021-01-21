"""
Module used to operate on notes
Saves information about training data - while finetunning can be used.
"""
import numpy as np


class NoteTokenizer:

    def __init__(self):
        self.notes_to_index = {}
        self.index_to_notes = {}
        self.unique_notes = 0

    def transform(self, list_array):
        """ Transform a list of note in string into index.

        :param: (list) list_array: list of note in string format
        :return: list of index of notes
        :rtype: list

        """
        transformed_list = []
        for instance in list_array:
            transformed_list.append([self.notes_to_index[note]
                                     for note in instance])
        return np.array(transformed_list, dtype=np.int32)

    def add_all_notes(self, notes):
        """ Adding all notes into the dictionary of the tokenizer

        :param: (list) notes : list of notes

        """
        for note in notes:
            # Have to turn array into string bc array type can't be key for map
            note_str = ','.join(str(a) for a in note)
            if note_str in self.notes_to_index:
                pass
            else:
                self.unique_notes += 1
                self.notes_to_index[note_str], self.index_to_notes[self.unique_notes] = self.unique_notes, note_str

    def add_empty_note(self, note):
        """ Add a new note into the dictionary

        :param: (str) note :  new note

        """
        assert note not in self.notes_to_index
        self.unique_notes += 1
        self.notes_to_index[note], self.index_to_notes[self.unique_notes] = self.unique_notes, note
