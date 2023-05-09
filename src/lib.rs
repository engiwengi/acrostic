#![allow(dead_code)]
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::{mpsc::channel, Arc},
};

use nanorand::{Rng, WyRand};
use parking_lot::Mutex;
use rayon::prelude::{ParallelBridge, ParallelIterator};
use ustr::Ustr;

#[derive(Debug)]
pub enum History {
    Minimized {
        slot_key: SlotKey,
        denied_words: Vec<Ustr>,
        previous_cells: Vec<MinimizedCell>,
    },
    ChoseRandomWord {
        slot_key: SlotKey,
        denied_words: Vec<Ustr>,
    },
    CompletedWord {
        slot_key: SlotKey,
        word: Ustr,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Direction {
    Across,
    Down,
}

impl Direction {
    fn other(self) -> Self {
        match self {
            Direction::Across => Direction::Down,
            Direction::Down => Direction::Across,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Slot {
    pub head: (i8, i8),
    pub length: usize,
    pub direction: Direction,
}

impl Slot {
    /// Checks if a cell index is inside this slot
    fn contains(&self, cell: (i8, i8)) -> bool {
        match self.direction {
            Direction::Across => cell.1 == self.head.1,
            Direction::Down => cell.0 == self.head.0,
        }
    }

    /// Checks if this slot overlaps another slot
    fn overlaps(self, other: Self) -> bool {
        self.iter().any(|s| other.contains(s))
    }

    // Iterates over the (x, y) index of each cell of this slot
    fn iter(self) -> SlotIterator {
        SlotIterator::new(self)
    }
}

pub struct SlotIterator {
    slot: Slot,
    current: usize,
}

impl SlotIterator {
    fn new(slot: Slot) -> Self {
        Self { slot, current: 0 }
    }
}

impl Iterator for SlotIterator {
    type Item = (i8, i8);

    fn next(&mut self) -> Option<Self::Item> {
        let (x, y) = self.slot.head;

        match self.slot.direction {
            Direction::Across if self.current <= self.slot.length => {
                let next = (x + self.current as i8, y);
                self.current += 1;
                Some(next)
            }
            Direction::Down if self.current <= self.slot.length => {
                let next = (x, y - self.current as i8);
                self.current += 1;
                Some(next)
            }
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SlotKey {
    index: usize,
    direction: Direction,
}

#[derive(Debug)]
pub struct MinimizedSlot {
    remaining: Vec<Ustr>,
    attempted: Vec<Ustr>,
    cells: Vec<Arc<Mutex<MinimizedCell>>>,
}

impl MinimizedSlot {
    /// Limits the remaining words to a randomly selected word returning the
    /// denied words.
    fn choose_random_word(&mut self, rand: &mut WyRand) -> Vec<Ustr> {
        debug_assert!(!self.remaining.is_empty());

        let chosen_word_index = rand.generate_range(0..self.remaining.len());
        self.remaining.swap(chosen_word_index, 0);
        self.remaining.split_off(1)
    }

    /// Restricts the cells of this slot to only allow letters of the remaining
    /// words in each cell. Returns the denied words and previous state of
    /// the cells
    fn minimize_cells(&mut self) -> (Vec<Ustr>, Vec<MinimizedCell>) {
        let mut allowed_chars = vec![MinimizedCell::empty(); self.cells.len()];
        let mut denied_words = Vec::new();

        let mut i = 0;
        while i < self.remaining.len() {
            let word = &self.remaining[i];
            if word
                .chars()
                .zip(self.cells.iter())
                .all(|(char, cell)| cell.lock().allows(char))
            {
                word.chars()
                    .zip(allowed_chars.iter_mut())
                    .for_each(|(char, allowed_chars)| {
                        allowed_chars.insert(char);
                    });

                i += 1;
            } else {
                let denied_word = self.remaining.swap_remove(i);
                denied_words.push(denied_word);
            }
        }
        (
            denied_words,
            allowed_chars
                .into_iter()
                .zip(self.cells.iter())
                .map(|(new_chars, cell)| std::mem::replace(&mut *cell.lock(), new_chars))
                .collect(),
        )
    }

    fn set_cells(&self, previous_cells: Vec<MinimizedCell>) {
        for (cell, previous) in self.cells.iter().zip(previous_cells.into_iter()) {
            *cell.lock() = previous;
        }
    }
}

pub struct Solver {
    pub grid: MinimizedGrid,
    pub rand: WyRand,
}

impl Solver {
    pub fn run(mut self) -> MinimizedGrid {
        let mut state = self.grid.choose_random_word(&mut self.rand);
        loop {
            state = match state {
                SolverState::ChooseRandomWord => self.grid.choose_random_word(&mut self.rand),
                SolverState::BackTracking => self.grid.backtrack(),
                SolverState::NoSolution(_) => panic!("no solution exists"),
                SolverState::Solved => return self.grid,
            };
        }
    }
}

#[derive(Debug)]
pub struct MinimizedGrid {
    history: Vec<History>,
    down_slots: Vec<MinimizedSlot>,
    across_slots: Vec<MinimizedSlot>,
    selected_words: HashMap<Ustr, SlotKey>,
    down_slots_overlapping_across_slots: Vec<Vec<Option<usize>>>,
    across_slots_overlapping_down_slots: Vec<Vec<Option<usize>>>,
}

impl MinimizedGrid {
    fn slot_mut(&mut self, key: SlotKey) -> &mut MinimizedSlot {
        match key.direction {
            Direction::Across => &mut self.across_slots[key.index],
            Direction::Down => &mut self.down_slots[key.index],
        }
    }

    fn slot(&self, key: SlotKey) -> &MinimizedSlot {
        match key.direction {
            Direction::Across => &self.across_slots[key.index],
            Direction::Down => &self.down_slots[key.index],
        }
    }

    fn down_slots_by_key(&self) -> impl Iterator<Item = (SlotKey, &'_ MinimizedSlot)> {
        self.down_slots.iter().enumerate().map(|(index, s)| {
            (
                SlotKey {
                    index,
                    direction: Direction::Down,
                },
                s,
            )
        })
    }

    fn across_slots_by_key(&self) -> impl Iterator<Item = (SlotKey, &'_ MinimizedSlot)> {
        self.across_slots.iter().enumerate().map(|(index, s)| {
            (
                SlotKey {
                    index,
                    direction: Direction::Across,
                },
                s,
            )
        })
    }

    fn most_constrained_slot_index(&self) -> Option<SlotKey> {
        self.down_slots_by_key()
            .chain(self.across_slots_by_key())
            .filter(|(_, slot)| slot.remaining.len() > 1)
            .min_by_key(|(_, slot)| slot.remaining.len())
            .map(|(key, _)| key)
    }

    fn choose_random_word(&mut self, rand: &mut WyRand) -> SolverState {
        let slot_key = match self.most_constrained_slot_index() {
            Some(slot_index) => slot_index,
            None => return SolverState::Solved,
        };

        let denied_words = self.slot_mut(slot_key).choose_random_word(rand);

        self.history.push(History::ChoseRandomWord {
            slot_key,
            denied_words,
        });

        let mut set = HashSet::with_capacity(1);
        set.insert(slot_key.index);

        self.minimize_slots(set, slot_key.direction)
    }

    fn minimize_slots(
        &mut self,
        mut slots_to_minimize: HashSet<usize>,
        mut direction: Direction,
    ) -> SolverState {
        while !slots_to_minimize.is_empty() {
            (slots_to_minimize, direction) =
                match self.minimize_slots_once(slots_to_minimize, direction) {
                    Ok(result) => (result, direction.other()),
                    Err(_e) => {
                        return SolverState::BackTracking;
                    }
                };
        }
        SolverState::ChooseRandomWord
    }

    fn minimize_slots_once(
        &mut self,
        mut slots_to_minimize: HashSet<usize>,
        direction: Direction,
    ) -> Result<HashSet<usize>, SolverError> {
        debug_assert!(!slots_to_minimize.is_empty());

        let iter = match direction {
            Direction::Across => self.across_slots.iter_mut().enumerate(),
            Direction::Down => self.down_slots.iter_mut().enumerate(),
        };

        let (s_history, r_history) = channel();
        let (s_abort, r_abort) = channel();
        let (s_next_slots, r_next_slots) = channel();

        iter.filter(|(slot_index, _)| slots_to_minimize.contains(slot_index))
            .par_bridge()
            .for_each_with(
                (s_history, s_abort, s_next_slots),
                |(s_history, s_abort, s_next_slots), (index, slot)| {
                    let (denied_words, previous_cells) = slot.minimize_cells();
                    match slot.remaining.first() {
                        None => {
                            s_abort.send(SolverError::NoPossibleWords).unwrap();
                        }
                        Some(word) if slot.remaining.len() == 1 => {
                            if !self.selected_words.contains_key(word) {
                                s_history
                                    .send(History::CompletedWord {
                                        word: *word,
                                        slot_key: SlotKey { index, direction },
                                    })
                                    .unwrap();
                            } else {
                                dbg!("duplicate word");
                                s_abort.send(SolverError::DuplicateWord).unwrap();
                            }
                        }
                        _ => {}
                    };
                    let send_next_slots = |opposite_slots_overlapping_cells: &[Option<usize>]| {
                        for opposite_slot_index in opposite_slots_overlapping_cells
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| slot.cells[*i].lock().ne(&previous_cells[*i]))
                            .filter_map(|(_, s)| *s)
                        {
                            s_next_slots.send(opposite_slot_index).unwrap();
                        }
                    };

                    match direction {
                        Direction::Across => {
                            send_next_slots(&self.across_slots_overlapping_down_slots[index]);
                        }
                        Direction::Down => {
                            send_next_slots(&self.down_slots_overlapping_across_slots[index]);
                        }
                    }

                    s_history
                        .send(History::Minimized {
                            slot_key: SlotKey { index, direction },
                            denied_words,
                            previous_cells,
                        })
                        .unwrap();
                },
            );

        slots_to_minimize.clear();
        slots_to_minimize.extend(r_next_slots.iter());

        for r in r_history.iter() {
            if let History::CompletedWord { word, slot_key } = r {
                self.selected_words.insert(word, slot_key);
            }
            self.history.push(r);
        }

        if let Some(e) = r_abort.iter().next() {
            return Err(e);
        }

        Ok(slots_to_minimize)
    }

    fn backtrack(&mut self) -> SolverState {
        while let Some(history) = self.history.pop() {
            match history {
                History::Minimized {
                    slot_key,
                    mut denied_words,
                    previous_cells,
                } => {
                    let slot = &mut self.slot_mut(slot_key);
                    slot.remaining.append(&mut denied_words);
                    slot.set_cells(previous_cells);
                }
                History::CompletedWord { word, .. } => {
                    self.selected_words.remove(&word);
                }
                History::ChoseRandomWord {
                    slot_key,
                    mut denied_words,
                } => {
                    let slot = &mut self.slot_mut(slot_key);
                    slot.remaining.append(&mut denied_words);

                    if slot.remaining.len() > 1 {
                        let bad_chosen_word = slot.remaining.swap_remove(0);
                        slot.attempted.push(bad_chosen_word);

                        return SolverState::ChooseRandomWord;
                    } else {
                        slot.remaining.append(&mut slot.attempted);
                    }
                }
            }
        }

        SolverState::NoSolution(SolverError::NoSolutionExists)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MinimizedCell {
    chars: HashSet<char>, // TODO represent as AtomicU32
}

impl Default for MinimizedCell {
    fn default() -> Self {
        Self {
            chars: ('A'..='Z').collect(),
        }
    }
}

impl MinimizedCell {
    fn empty() -> Self {
        Self {
            chars: Default::default(),
        }
    }

    fn allows(&self, char: char) -> bool {
        // self.chars.get_bit(char as usize, Ordering::SeqCst)
        self.chars.contains(&char)
    }

    fn insert(&mut self, char: char) {
        // self.chars.set_bit(char as usize, Ordering::SeqCst)
        self.chars.insert(char);
    }

    fn swap(&mut self, chars: HashSet<char>) -> HashSet<char> {
        // self.chars.swap(chars, Ordering::SeqCst)
        std::mem::replace(&mut self.chars, chars)
    }
}

#[derive(Clone)]
enum SolverState {
    NoSolution(SolverError),
    Solved,
    ChooseRandomWord,
    BackTracking,
}

#[derive(Debug, Clone)]
pub enum SolverError {
    NoPossibleWords,
    RuntimeError(String),
    NoSolutionExists,
    DuplicateWord,
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn slot_overlap() {
        let len2 = |head, direction| Slot {
            head,
            length: 2,
            direction,
        };
        let top_across = len2((0, 2), Direction::Across);
        let mid_across = len2((0, 1), Direction::Across);
        let bot_across = len2((0, 0), Direction::Across);

        let left_down = len2((0, 2), Direction::Down);
        let mid_down = len2((1, 2), Direction::Down);
        let right_down = len2((2, 2), Direction::Down);

        assert!(top_across.overlaps(left_down));
        assert!(top_across.overlaps(mid_down));
        assert!(top_across.overlaps(right_down));

        assert!(top_across.overlaps(top_across));
        assert!(!top_across.overlaps(mid_across));
        assert!(!top_across.overlaps(bot_across));

        assert!(mid_across.overlaps(left_down));
        assert!(mid_across.overlaps(mid_down));
        assert!(mid_across.overlaps(right_down));

        assert!(!mid_across.overlaps(top_across));
        assert!(mid_across.overlaps(mid_across));
        assert!(!mid_across.overlaps(bot_across));

        assert!(bot_across.overlaps(left_down));
        assert!(bot_across.overlaps(mid_down));
        assert!(bot_across.overlaps(right_down));

        assert!(!bot_across.overlaps(top_across));
        assert!(!bot_across.overlaps(mid_across));
        assert!(bot_across.overlaps(bot_across));

        assert!(left_down.overlaps(top_across));
        assert!(left_down.overlaps(mid_across));
        assert!(left_down.overlaps(bot_across));

        assert!(left_down.overlaps(left_down));
        assert!(!left_down.overlaps(mid_down));
        assert!(!left_down.overlaps(right_down));

        assert!(mid_down.overlaps(top_across));
        assert!(mid_down.overlaps(mid_across));
        assert!(mid_down.overlaps(bot_across));

        assert!(!mid_down.overlaps(left_down));
        assert!(mid_down.overlaps(mid_down));
        assert!(!mid_down.overlaps(right_down));

        assert!(right_down.overlaps(top_across));
        assert!(right_down.overlaps(mid_across));
        assert!(right_down.overlaps(bot_across));

        assert!(!right_down.overlaps(left_down));
        assert!(!right_down.overlaps(mid_down));
        assert!(right_down.overlaps(right_down));
    }
}
