use std::{borrow::Borrow, convert::TryFrom};

use ascii::{AsAsciiStrError, AsciiChar, AsciiStr, AsciiString};
use enumset::EnumSet;
use fst::Automaton;

use crate::cw::AsciiCharInternal;

pub struct Intersections<S>(pub Vec<Box<dyn Automaton<State = S>>>);

impl<S> Automaton for Intersections<S> {
    type State = Vec<S>;

    #[inline(always)]
    fn start(&self) -> Self::State {
        self.0.iter().map(|a| a.start()).collect()
    }

    #[inline(always)]
    fn is_match(&self, state: &Self::State) -> bool {
        state.iter().zip(self.0.iter()).all(|(s, a)| a.is_match(s))
    }

    #[inline(always)]
    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        state
            .iter()
            .zip(self.0.iter())
            .map(|(s, a)| a.accept(s, byte))
            .collect()
    }

    #[inline(always)]
    fn can_match(&self, state: &Self::State) -> bool {
        state.iter().zip(self.0.iter()).all(|(s, a)| a.can_match(s))
    }

    #[inline(always)]
    fn will_always_match(&self, state: &Self::State) -> bool {
        state
            .iter()
            .zip(self.0.iter())
            .all(|(s, a)| a.will_always_match(s))
    }

    #[inline(always)]
    fn accept_eof(&self, state: &Self::State) -> Option<Self::State> {
        state
            .iter()
            .zip(self.0.iter())
            .map(|(s, a)| a.accept_eof(s))
            .collect()
    }
}

pub struct CharAt {
    char: EnumSet<AsciiCharInternal>, // TODO use an enumset
    idx: usize,
}

impl CharAt {
    pub fn new(set: EnumSet<AsciiCharInternal>, idx: usize) -> Self {
        Self { char: set, idx }
    }
}

impl Automaton for CharAt {
    type State = Option<usize>;

    #[inline(always)]
    fn start(&self) -> Self::State {
        Some(0)
    }

    #[inline(always)]
    fn is_match(&self, state: &Self::State) -> bool {
        self.will_always_match(state)
    }

    #[inline(always)]
    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        match state {
            Some(pos) if *pos == self.idx => {
                if let Ok(char) = AsciiChar::from_ascii(byte) {
                    Some(pos + 1).filter(|_| self.char.contains(AsciiCharInternal::from(char)))
                } else {
                    None
                }
            }
            Some(pos) => Some(pos + 1),
            None => None,
        }
    }

    #[inline(always)]
    fn will_always_match(&self, state: &Self::State) -> bool {
        state.filter(|pos| *pos >= self.idx).is_some()
    }
    #[inline(always)]
    fn can_match(&self, state: &Self::State) -> bool {
        state.is_some()
    }
}

pub struct Length {
    size: usize,
}

impl Length {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Automaton for Length {
    type State = Option<usize>;

    #[inline(always)]
    fn start(&self) -> Self::State {
        Some(0)
    }

    #[inline(always)]
    fn is_match(&self, state: &Self::State) -> bool {
        Some(self.size) == *state
    }

    #[inline(always)]
    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        if let Some(pos) = *state {
            Some(pos + 1).filter(|pos| *pos <= self.size)
        } else {
            None
        }
    }

    #[inline(always)]
    fn can_match(&self, state: &Self::State) -> bool {
        state.is_some()
    }
}

pub struct VaguePattern<'a> {
    query: &'a [Option<EnumSet<AsciiCharInternal>>],
}

impl<'a> VaguePattern<'a> {
    pub fn new(query: &'a [Option<EnumSet<AsciiCharInternal>>]) -> Self {
        Self { query }
    }
}

impl<'a> Automaton for VaguePattern<'a> {
    type State = Option<usize>;

    #[inline(always)]
    fn start(&self) -> Self::State {
        Some(0)
    }

    #[inline(always)]
    fn is_match(&self, state: &Self::State) -> bool {
        *state == Some(self.query.len())
    }

    #[inline(always)]
    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        if let Ok(char) = AsciiChar::from_ascii(byte) {
            let internal_char = AsciiCharInternal::from(char);
            if let Some(pos) = *state {
                // and there is still a matching byte at the current position...
                let match_char = self.query.get(pos).copied();
                if let Some(match_char) = match_char {
                    if let Some(match_set) = match_char {
                        return Some(pos + 1).filter(|_| match_set.contains(internal_char));
                    } else {
                        return Some(pos + 1);
                    }
                }
            }
        }
        // otherwise we're either past the end or didn't match the byte
        None
    }

    #[inline(always)]
    fn can_match(&self, state: &Self::State) -> bool {
        state.is_some()
    }
}

pub struct WildCardString {
    query: AsciiString,
}

impl From<AsciiString> for WildCardString {
    fn from(query: AsciiString) -> Self {
        Self { query }
    }
}

impl Automaton for WildCardString {
    type State = <WildCardStr<'static> as Automaton>::State;

    fn start(&self) -> Self::State {
        WildCardStr::from(self.query.borrow()).start()
    }

    fn is_match(&self, state: &Self::State) -> bool {
        WildCardStr::from(self.query.borrow()).is_match(state)
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        WildCardStr::from(self.query.borrow()).accept(state, byte)
    }
}

pub struct WildCardStr<'a> {
    query: &'a [AsciiChar],
}

impl<'a> From<&'a AsciiStr> for WildCardStr<'a> {
    fn from(ascii_str: &'a AsciiStr) -> Self {
        Self {
            query: ascii_str.as_slice(),
        }
    }
}
impl<'a> TryFrom<&'a str> for WildCardStr<'a> {
    type Error = AsAsciiStrError;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        Ok(Self {
            query: AsciiStr::from_ascii(value)?.as_slice(),
        })
    }
}

impl<'a> Automaton for WildCardStr<'a> {
    type State = Option<usize>;

    fn start(&self) -> Self::State {
        Some(0)
    }

    fn is_match(&self, state: &Self::State) -> bool {
        *state == Some(self.query.len())
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        if let Ok(char) = AsciiChar::from_ascii(byte) {
            if let Some(pos) = *state {
                // and there is still a matching byte at the current position...
                let match_char = self.query.get(pos).copied();
                if match_char == Some(char) || match_char == Some(AsciiChar::Question) {
                    // then move forward
                    return Some(pos + 1);
                }
            }
        }
        // otherwise we're either past the end or didn't match the byte
        None
    }

    fn can_match(&self, state: &Self::State) -> bool {
        state.is_some()
    }
}
