/*
File: offline/readback.rs

Purpose:
Getting pixels back from the GPU: the row-padded layout of a texture -> buffer copy, a future
that resolves when `map_async` completes, and (native only) a dependency-free `block_on` used
by the blocking entry points.

Key structures:
- `RowLayout`: tight vs `COPY_BYTES_PER_ROW_ALIGNMENT`-padded row sizes of one copy.
- `MapFuture`: resolves with the `map_async` result.

Key functions:
- `RowLayout::new()`, `map_read()`, `block_on()` (native).

Notes:
On native wgpu, `map_async` completes during `Device::poll` (or a later `Queue::submit`); on the
web the browser completes it. The blocking entry points poll with `PollType::Wait` first, so
their future is already resolved when `block_on` polls it.
*/

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};

use super::error::OfflineError;

/// Row sizes of a `width x height` RGBA8 texture copied into a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowLayout {
    /// `width * 4`: bytes of pixel data per row.
    pub tight: u32,
    /// `tight` rounded up to `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (the buffer's row pitch).
    pub padded: u32,
    /// Number of rows.
    pub rows: u32,
}

impl RowLayout {
    /// The layout of a `width x height` RGBA8 copy.
    ///
    /// # Errors
    /// [`OfflineError::SizeOverflow`] when a row or the buffer does not fit `u32` / `u64`.
    pub(crate) fn new(width: u32, height: u32) -> Result<Self, OfflineError> {
        let overflow = OfflineError::SizeOverflow { what: "the readback buffer" };
        let tight = width.checked_mul(4).ok_or(OfflineError::SizeOverflow { what: "the readback buffer" })?;
        let padded = tight.checked_next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT).ok_or(overflow)?;
        Ok(Self { tight, padded, rows: height })
    }

    /// Size of the buffer receiving the copy, in bytes.
    #[must_use]
    pub(crate) fn buffer_size(&self) -> u64 {
        u64::from(self.padded) * u64::from(self.rows)
    }
}

/// Shared between a pending map and its callback.
#[derive(Debug, Default)]
struct MapState {
    /// The callback's result, once it ran.
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    /// The task to wake when it does.
    waker: Option<Waker>,
}

/// Resolves when the `map_async` started by [`map_read`] completes.
#[derive(Debug)]
pub(crate) struct MapFuture {
    /// Written by the map callback.
    state: Arc<Mutex<MapState>>,
}

impl Future for MapFuture {
    type Output = Result<(), OfflineError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // A poisoned lock only means a panic elsewhere while holding it; the state stays valid.
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(result) = state.result.take() {
            return Poll::Ready(result.map_err(|error| OfflineError::Readback(error.to_string())));
        }
        state.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// Starts mapping `slice` for reading and returns the future of its completion.
pub(crate) fn map_read(slice: &wgpu::BufferSlice<'_>) -> MapFuture {
    let state = Arc::new(Mutex::new(MapState::default()));
    let shared = Arc::clone(&state);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let waker = {
            let mut state = shared.lock().unwrap_or_else(PoisonError::into_inner);
            state.result = Some(result);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    });
    MapFuture { state }
}

/// Wakes a parked thread.
#[cfg(not(target_arch = "wasm32"))]
struct ThreadWaker(std::thread::Thread);

#[cfg(not(target_arch = "wasm32"))]
impl std::task::Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Runs `future` to completion on the current thread, parking it while the future is pending.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::park();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_padded_to_the_copy_alignment() {
        assert_eq!(RowLayout::new(64, 3).ok(), Some(RowLayout { tight: 256, padded: 256, rows: 3 }));
        assert_eq!(RowLayout::new(65, 2).ok(), Some(RowLayout { tight: 260, padded: 512, rows: 2 }));
        assert_eq!(RowLayout::new(1, 1).map(|layout| layout.buffer_size()).ok(), Some(256));
        assert!(RowLayout::new(u32::MAX, 1).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn block_on_waits_for_a_wake_from_another_thread() {
        let state = Arc::new(Mutex::new(MapState::default()));
        let future = MapFuture { state: Arc::clone(&state) };
        let completer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let waker = {
                let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
                state.result = Some(Ok(()));
                state.waker.take()
            };
            if let Some(waker) = waker {
                waker.wake();
            }
        });
        assert!(block_on(future).is_ok());
        assert!(completer.join().is_ok());
    }
}
