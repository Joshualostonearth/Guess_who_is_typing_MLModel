/* ============================================================
   ChronoType — script.js
   Core typing-test logic with comprehensive keystroke biometric
   data collection for ML research.

   Captures:
     1. Key Timing  — dwell, flight, DD, UU, digraph, trigraph
     2. Behavior    — shift pref, backspace style, rollover, caps
     3. Aggregates  — WPM, error rate, burstiness, fatigue curve
   ============================================================ */

(() => {
  'use strict';

  // ——————————————— Word Bank (~200 common words) ———————————————
  const WORD_BANK = [
    'apple','system','logic','focus','brain','crane','drive','earth',
    'flame','grape','house','image','judge','knife','lemon','mango',
    'night','ocean','piano','queen','river','stone','table','unity',
    'vivid','water','xenon','yacht','zebra','above','below','chair',
    'dance','elite','frost','giant','happy','intro','jolly','karma',
    'light','magic','noble','orbit','proud','quick','royal','solar',
    'trade','urban','value','world','array','blend','cloud','digit',
    'entry','field','graph','hover','input','joint','layer','model',
    'nerve','outer','pixel','query','route','shift','token','ultra',
    'voice','width','yield','alpha','board','cycle','delta','event',
    'frame','guide','index','jewel','known','learn','lunar','media',
    'north','opera','panel','radar','scale','theme','under','valid',
    'watch','acute','blank','climb','draft','exact','flash','grace',
    'human','ivory','joker','knack','level','march','noted','offer',
    'place','quest','ratio','spine','trail','usher','vapor','whale',
    'about','basic','close','debug','equal','final','globe','heart',
    'ident','joust','keeps','limit','minor','novel','occur','phase',
    'quiet','range','share','think','unity','verse','waste','extra',
    'young','zones','birth','cover','depth','epoch','fiber','green',
    'hatch','inner','juice','kicks','logic','metal','nexus','oxide',
    'power','quote','realm','sweep','tight','usage','vital','wreck',
    'bread','click','delay','error','fetch','grain','helix','issue',
    'jumps','knots','lease','multi','noted','omega','print','relay',
    'state','tower','union','vault','wired','amber','brace','charm',
    'drift','evade','flint','glyph','haste','incur','jazzy','kneel',
    'lyric','mocha','noisy','optic','plumb','quilt','ridge','shade',
    'twist','umbra','vigor','whisk','oxide','bliss','crest','dwell'
  ];

  // Common digraphs and trigraphs for latency tracking
  const TRACKED_DIGRAPHS = [
    'th','he','in','er','an','re','on','at','en','nd',
    'ti','es','or','te','of','ed','is','it','al','ar',
    'st','to','nt','ng','se','ha','as','ou','io','le',
    've','co','me','de','hi','ri','ro','ic','ne','ea',
    'ra','ce'
  ];
  const TRACKED_TRIGRAPHS = [
    'the','ing','and','ion','her','for','tha','ith',
    'hat','ent','thi','ter','ere','not','all','was',
    'are','but','con','wit'
  ];

  // ——————————————— Configuration ———————————————
  const WORD_COUNT         = 30;    // words per test
  const TEST_TIME          = 15;    // seconds
  const WPM_WINDOW_SIZE   = 5;     // seconds per WPM window for fatigue curve
  const BACKSPACE_HOLD_THRESHOLD = 80; // ms — below this = "held" backspace

  // ——————————————— DOM References ———————————————
  const $wordsContainer = document.getElementById('words-container');
  const $timerDisplay   = document.getElementById('timer-display');
  const $wpmDisplay     = document.getElementById('wpm-display');
  const $typingArea     = document.getElementById('typing-area');
  const $resultsScreen  = document.getElementById('results-screen');
  const $restartBtn     = document.getElementById('restart-btn');
  const $restartHint    = document.getElementById('restart-hint');
  const $header         = document.getElementById('header');



  // Result value elements
  const $finalWpm         = document.getElementById('final-wpm');
  const $finalAccuracy    = document.getElementById('final-accuracy');
  const $finalChars       = document.getElementById('final-chars');
  const $finalErrors      = document.getElementById('final-errors');
  const $finalRegressions = document.getElementById('final-regressions');
  const $finalTime        = document.getElementById('final-time');
  const $finalDwell       = document.getElementById('final-dwell');
  const $finalFlight      = document.getElementById('final-flight');
  const $finalBurstiness  = document.getElementById('final-burstiness');
  const $finalRollovers   = document.getElementById('final-rollovers');
  const $finalShiftPref   = document.getElementById('final-shift-pref');
  const $finalErrorRate   = document.getElementById('final-error-rate');
  const $saveStatus       = document.getElementById('save-status');

  const $svmSection    = document.getElementById('svm-section');
  const $svmLoading    = document.getElementById('svm-loading');
  const $svmGuess      = document.getElementById('svm-guess');
  const $svmPrediction = document.getElementById('svm-prediction');
  const $svmSubtext    = document.getElementById('svm-subtext');
  const $svmYesBtn     = document.getElementById('svm-yes-btn');
  const $svmNoBtn      = document.getElementById('svm-no-btn');

  // ——————————————— State ———————————————
  let playerName     = 'Unknown';    // default typer's name
  let chars          = [];    // flat array of all <span class="char"> elements
  let caretIndex     = 0;     // current position in chars[]
  let timerStarted   = false;
  let timerInterval  = null;
  let timeRemaining  = TEST_TIME;
  let testActive     = false;
  let testFinished   = false;
  let nameScreenActive = false;

  let pendingSessionData  = null;
  let svmPredictedName    = null; // name screen is showing

  // Basic metrics
  let correctCount     = 0;
  let errorCount       = 0;
  let regressionCount  = 0;
  let totalTyped       = 0;   // every forward key-press (correct + error)

  // ——————————————— Keystroke Collector ———————————————

  /**
   * KeystrokeCollector — Records raw keydown/keyup events with
   * high-resolution timestamps and computes biometric features.
   */
  class KeystrokeCollector {
    constructor() {
      this.reset();
    }

    reset() {
      // Raw event log
      this.rawEvents = [];        // { type, key, code, location, timestamp, charIndex }

      // Currently held keys (for rollover detection)
      this.pressedKeys = new Map(); // code → { key, timestamp, location }

      // Ordered press/release arrays for timing computation
      this.presses  = [];  // { key, code, timestamp, location, charIndex }
      this.releases = [];  // { key, code, timestamp, location }

      // Behavioral counters
      this.leftShiftCount  = 0;
      this.rightShiftCount = 0;
      this.capsLockCount   = 0;
      this.ctrlACount      = 0;
      this.backspaceTimestamps = []; // timestamps of each backspace press

      // Rollover tracking
      this.rolloverEvents = [];  // { key1, key2, overlapMs }

      // WPM windowed tracking for fatigue curve
      this.windowedCorrectCounts = []; // chars correct in each window
      this.windowStart = null;
      this.currentWindowChars = 0;
    }

    /**
     * Record a keydown event. Called from the global keydown listener.
     */
    recordKeyDown(e, charIndex) {
      const timestamp = performance.now();

      const event = {
        type: 'keydown',
        key: e.key,
        code: e.code,
        location: e.location,
        timestamp,
        charIndex
      };
      this.rawEvents.push(event);

      // ——— Rollover detection ———
      // If there are already keys held down, this is a rollover
      if (this.pressedKeys.size > 0 && e.key.length === 1) {
        for (const [heldCode, heldInfo] of this.pressedKeys) {
          // Only count rollover between printable keys
          if (heldInfo.key.length === 1) {
            const overlapMs = timestamp - heldInfo.timestamp;
            this.rolloverEvents.push({
              key1: heldInfo.key,
              key2: e.key,
              overlapMs
            });
          }
        }
      }

      // Track held keys
      this.pressedKeys.set(e.code, { key: e.key, timestamp, location: e.location });

      // ——— Modifier / special key tracking ———
      if (e.key === 'Shift') {
        if (e.location === 1) this.leftShiftCount++;
        else if (e.location === 2) this.rightShiftCount++;
      }

      if (e.key === 'CapsLock') {
        this.capsLockCount++;
      }

      if (e.key === 'a' && e.ctrlKey) {
        this.ctrlACount++;
      }

      if (e.key === 'Backspace') {
        this.backspaceTimestamps.push(timestamp);
      }

      // Record ordered press (for printable + space + backspace)
      this.presses.push({ key: e.key, code: e.code, timestamp, location: e.location, charIndex });
    }

    /**
     * Record a keyup event. Called from the global keyup listener.
     */
    recordKeyUp(e) {
      const timestamp = performance.now();

      const event = {
        type: 'keyup',
        key: e.key,
        code: e.code,
        location: e.location,
        timestamp
      };
      this.rawEvents.push(event);

      // Remove from held keys
      this.pressedKeys.delete(e.code);

      // Record ordered release
      this.releases.push({ key: e.key, code: e.code, timestamp, location: e.location });
    }

    /**
     * Track a correct character for WPM windowing.
     */
    recordCorrectChar() {
      this.currentWindowChars++;
    }

    /**
     * Advance the WPM window. Called every WPM_WINDOW_SIZE seconds.
     */
    snapshotWindow() {
      this.windowedCorrectCounts.push(this.currentWindowChars);
      this.currentWindowChars = 0;
    }

    // =============================================
    //  POST-PROCESSING: Compute all derived features
    // =============================================

    computeAllFeatures(elapsedSeconds) {
      const features = {};

      // ——— 1. Key Timing (Temporal Features) ———
      features.timing = this._computeTimingFeatures();

      // ——— 2. Behavioral Patterns ———
      features.behavior = this._computeBehavioralFeatures();

      // ——— 3. Aggregate Performance ———
      features.aggregate = this._computeAggregateFeatures(elapsedSeconds);

      // ——— 4. Raw data for JSON export ———
      features.raw = {
        events: this.rawEvents,
        presses: this.presses,
        releases: this.releases,
        rolloverEvents: this.rolloverEvents
      };

      return features;
    }

    // ——— Timing Features ———

    _computeTimingFeatures() {
      const timing = {
        dwellTimes: [],
        flightTimes: [],
        ddTimes: [],
        uuTimes: [],
        digraphLatencies: {},
        trigraphLatencies: {}
      };

      // Build a map of code → releases for dwell time matching
      // We need to match each keydown with its corresponding keyup
      const pressReleasePairs = this._matchPressRelease();

      // Dwell times
      timing.dwellTimes = pressReleasePairs
        .map(pair => ({
          key: pair.press.key,
          dwellMs: pair.release.timestamp - pair.press.timestamp
        }))
        .filter(d => d.dwellMs > 0 && d.dwellMs < 2000); // sanity filter

      // Flight times (time from release[n] to press[n+1])
      // Only for consecutive printable character presses
      const printablePresses = this.presses.filter(p => p.key.length === 1 || p.key === ' ');
      const printableReleases = this._getOrderedPrintableReleases();

      for (let i = 0; i < printableReleases.length - 1; i++) {
        const releaseTime = printableReleases[i].timestamp;
        // Find the next printable press after this release
        const nextPress = printablePresses.find(p => p.timestamp > releaseTime - 1);
        if (nextPress && nextPress !== printablePresses[0]) {
          const idx = printablePresses.indexOf(nextPress);
          if (idx > 0) {
            const flightMs = nextPress.timestamp - releaseTime;
            if (Math.abs(flightMs) < 2000) { // sanity
              timing.flightTimes.push({
                from: printablePresses[idx - 1]?.key || '?',
                to: nextPress.key,
                flightMs
              });
            }
          }
        }
      }

      // Simpler approach for flight, DD, UU: use ordered printable presses/releases
      const orderedPresses = printablePresses.sort((a, b) => a.timestamp - b.timestamp);

      // Down-to-Down times
      for (let i = 1; i < orderedPresses.length; i++) {
        const ddMs = orderedPresses[i].timestamp - orderedPresses[i - 1].timestamp;
        if (ddMs > 0 && ddMs < 2000) {
          timing.ddTimes.push({
            from: orderedPresses[i - 1].key,
            to: orderedPresses[i].key,
            ddMs
          });
        }
      }

      // Up-to-Up times
      const orderedReleases = printableReleases.sort((a, b) => a.timestamp - b.timestamp);
      for (let i = 1; i < orderedReleases.length; i++) {
        const uuMs = orderedReleases[i].timestamp - orderedReleases[i - 1].timestamp;
        if (uuMs > 0 && uuMs < 2000) {
          timing.uuTimes.push({
            from: orderedReleases[i - 1].key,
            to: orderedReleases[i].key,
            uuMs
          });
        }
      }

      // Flight times (recalculated cleanly: release[i] → press[i+1])
      timing.flightTimes = [];
      for (let i = 0; i < orderedPresses.length - 1; i++) {
        // Find the release for orderedPresses[i]
        const releaseForPress = pressReleasePairs.find(
          p => p.press.timestamp === orderedPresses[i].timestamp
        );
        if (releaseForPress) {
          const flightMs = orderedPresses[i + 1].timestamp - releaseForPress.release.timestamp;
          if (Math.abs(flightMs) < 2000) {
            timing.flightTimes.push({
              from: orderedPresses[i].key,
              to: orderedPresses[i + 1].key,
              flightMs
            });
          }
        }
      }

      // Digraph latencies (DD-time for specific 2-letter combos)
      for (const dd of timing.ddTimes) {
        const digraph = (dd.from + dd.to).toLowerCase();
        if (TRACKED_DIGRAPHS.includes(digraph)) {
          if (!timing.digraphLatencies[digraph]) timing.digraphLatencies[digraph] = [];
          timing.digraphLatencies[digraph].push(dd.ddMs);
        }
      }

      // Trigraph latencies
      for (let i = 2; i < orderedPresses.length; i++) {
        const trigraph = (
          orderedPresses[i - 2].key +
          orderedPresses[i - 1].key +
          orderedPresses[i].key
        ).toLowerCase();

        if (TRACKED_TRIGRAPHS.includes(trigraph)) {
          const totalMs = orderedPresses[i].timestamp - orderedPresses[i - 2].timestamp;
          if (totalMs > 0 && totalMs < 3000) {
            if (!timing.trigraphLatencies[trigraph]) timing.trigraphLatencies[trigraph] = [];
            timing.trigraphLatencies[trigraph].push(totalMs);
          }
        }
      }

      // Compute averages
      timing.avgDwellMs  = this._avg(timing.dwellTimes.map(d => d.dwellMs));
      timing.avgFlightMs = this._avg(timing.flightTimes.map(d => d.flightMs));
      timing.avgDDMs     = this._avg(timing.ddTimes.map(d => d.ddMs));
      timing.avgUUMs     = this._avg(timing.uuTimes.map(d => d.uuMs));

      return timing;
    }

    /**
     * Match each keydown with its next corresponding keyup (by code).
     */
    _matchPressRelease() {
      const pairs = [];
      const usedReleases = new Set();

      for (const press of this.presses) {
        // Find the earliest matching release after this press
        for (let i = 0; i < this.releases.length; i++) {
          if (
            !usedReleases.has(i) &&
            this.releases[i].code === press.code &&
            this.releases[i].timestamp >= press.timestamp
          ) {
            pairs.push({ press, release: this.releases[i] });
            usedReleases.add(i);
            break;
          }
        }
      }

      return pairs;
    }

    /**
     * Get releases for printable keys, ordered by timestamp.
     */
    _getOrderedPrintableReleases() {
      return this.releases
        .filter(r => r.key.length === 1 || r.key === ' ')
        .sort((a, b) => a.timestamp - b.timestamp);
    }

    // ——— Behavioral Features ———

    _computeBehavioralFeatures() {
      const behavior = {};

      // Modifier key preference
      behavior.leftShiftCount  = this.leftShiftCount;
      behavior.rightShiftCount = this.rightShiftCount;
      const totalShifts = this.leftShiftCount + this.rightShiftCount;
      behavior.shiftPreference = totalShifts === 0
        ? 'none'
        : this.leftShiftCount > this.rightShiftCount
          ? 'left'
          : this.leftShiftCount < this.rightShiftCount
            ? 'right'
            : 'balanced';

      // Backspace / correction style
      behavior.backspaceCount = this.backspaceTimestamps.length;
      behavior.backspaceIntervals = [];
      let holdCount = 0;
      let tapCount  = 0;
      for (let i = 1; i < this.backspaceTimestamps.length; i++) {
        const interval = this.backspaceTimestamps[i] - this.backspaceTimestamps[i - 1];
        behavior.backspaceIntervals.push(interval);
        if (interval < BACKSPACE_HOLD_THRESHOLD) holdCount++;
        else tapCount++;
      }
      behavior.backspaceStyle = behavior.backspaceCount <= 1
        ? 'minimal'
        : holdCount > tapCount
          ? 'held'
          : tapCount > holdCount
            ? 'tapped'
            : 'mixed';

      // Ctrl+A usage
      behavior.ctrlACount = this.ctrlACount;

      // Caps Lock
      behavior.capsLockCount = this.capsLockCount;

      // Key rollover
      behavior.rolloverCount = this.rolloverEvents.length;
      behavior.avgRolloverDurationMs = this._avg(
        this.rolloverEvents.map(r => r.overlapMs)
      );
      behavior.rolloverDetails = this.rolloverEvents;

      return behavior;
    }

    // ——— Aggregate Features ———

    _computeAggregateFeatures(elapsedSeconds) {
      const aggregate = {};

      // WPM
      aggregate.wpm = elapsedSeconds > 0
        ? Math.round((correctCount / 5) / (elapsedSeconds / 60))
        : 0;

      // Error rate
      aggregate.errorRate = totalTyped > 0
        ? parseFloat(((errorCount / totalTyped) * 100).toFixed(2))
        : 0;

      // Burstiness: coefficient of variation of inter-key intervals
      const printablePresses = this.presses
        .filter(p => p.key.length === 1 || p.key === ' ')
        .sort((a, b) => a.timestamp - b.timestamp);

      const intervals = [];
      for (let i = 1; i < printablePresses.length; i++) {
        const dt = printablePresses[i].timestamp - printablePresses[i - 1].timestamp;
        if (dt > 0 && dt < 5000) intervals.push(dt);
      }

      const meanInterval = this._avg(intervals);
      const stdInterval  = this._std(intervals);
      aggregate.burstiness = meanInterval > 0
        ? parseFloat((stdInterval / meanInterval).toFixed(3))
        : 0;

      // Fatigue curve: WPM per window
      // Ensure the last partial window is captured
      if (this.currentWindowChars > 0) {
        this.windowedCorrectCounts.push(this.currentWindowChars);
      }

      aggregate.wpmOverTime = this.windowedCorrectCounts.map(chars =>
        Math.round((chars / 5) / (WPM_WINDOW_SIZE / 60))
      );

      // Speed decay rate: slope of linear regression on wpmOverTime
      aggregate.speedDecayRate = this._linearSlope(aggregate.wpmOverTime);

      // Session duration
      aggregate.sessionDurationSeconds = elapsedSeconds;

      return aggregate;
    }

    // ——— Math Utilities ———

    _avg(arr) {
      if (!arr.length) return 0;
      return parseFloat((arr.reduce((s, v) => s + v, 0) / arr.length).toFixed(2));
    }

    _std(arr) {
      if (arr.length < 2) return 0;
      const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
      const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
      return Math.sqrt(variance);
    }

    _linearSlope(values) {
      if (values.length < 2) return 0;
      const n = values.length;
      let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
      for (let i = 0; i < n; i++) {
        sumX  += i;
        sumY  += values[i];
        sumXY += i * values[i];
        sumXX += i * i;
      }
      const denom = n * sumXX - sumX * sumX;
      if (denom === 0) return 0;
      return parseFloat(((n * sumXY - sumX * sumY) / denom).toFixed(3));
    }
  }

  // ——————————————— Collector Instance ———————————————
  let collector = new KeystrokeCollector();



  // ——————————————— Utilities ———————————————

  /** Pick `n` random words from the word bank (allows repeats). */
  function pickWords(n) {
    const words = [];
    for (let i = 0; i < n; i++) {
      words.push(WORD_BANK[Math.floor(Math.random() * WORD_BANK.length)]);
    }
    return words;
  }

  /** Build the DOM for the typing area and populate `chars[]`. */
  function renderWords(words) {
    $wordsContainer.innerHTML = '';
    chars = [];

    words.forEach((word, wIdx) => {
      const wordSpan = document.createElement('span');
      wordSpan.className = 'word';

      // Wrap each character
      for (const ch of word) {
        const charSpan = document.createElement('span');
        charSpan.className = 'char';
        charSpan.textContent = ch;
        charSpan.dataset.char = ch;
        wordSpan.appendChild(charSpan);
        chars.push(charSpan);
      }

      // After every word except the last, add a space character
      if (wIdx < words.length - 1) {
        const spaceSpan = document.createElement('span');
        spaceSpan.className = 'char';
        spaceSpan.textContent = '\u00A0';   // non-breaking space for visibility
        spaceSpan.dataset.char = ' ';
        wordSpan.appendChild(spaceSpan);
        chars.push(spaceSpan);
      }

      $wordsContainer.appendChild(wordSpan);
    });

    // Place the initial caret
    if (chars.length) chars[0].classList.add('current');
  }

  // ——————————————— Timer ———————————————

  let wpmWindowTimer = null; // interval for fatigue windowing

  function startTimer() {
    timerStarted = true;
    testActive   = true;
    timeRemaining = TEST_TIME;
    $timerDisplay.textContent = timeRemaining;

    collector.windowStart = performance.now();

    timerInterval = setInterval(() => {
      timeRemaining--;
      $timerDisplay.textContent = timeRemaining;

      // Update live WPM every second
      updateLiveWpm();

      if (timeRemaining <= 0) {
        endTest();
      }
    }, 1000);

    // Fatigue window snapshots
    wpmWindowTimer = setInterval(() => {
      collector.snapshotWindow();
    }, WPM_WINDOW_SIZE * 1000);
  }

  function updateLiveWpm() {
    const elapsed = TEST_TIME - timeRemaining;
    if (elapsed <= 0) { $wpmDisplay.textContent = 0; return; }
    // Standard WPM: 1 word = 5 characters
    const wpm = Math.round((correctCount / 5) / (elapsed / 60));
    $wpmDisplay.textContent = wpm;
  }

  // ——————————————— End Test & Show Results ———————————————

  function endTest() {
    clearInterval(timerInterval);
    clearInterval(wpmWindowTimer);
    testActive   = false;
    testFinished = true;

    const elapsed = TEST_TIME - timeRemaining;

    const features = collector.computeAllFeatures(elapsed);

    const wpm = elapsed > 0
      ? Math.round((correctCount / 5) / (elapsed / 60))
      : 0;

    const accuracy = totalTyped > 0
      ? Math.round(((totalTyped - errorCount) / totalTyped) * 100)
      : 100;

    $finalWpm.textContent         = wpm;
    $finalAccuracy.textContent    = accuracy + '%';
    $finalChars.textContent       = totalTyped;
    $finalErrors.textContent      = errorCount;
    $finalRegressions.textContent = regressionCount;
    $finalTime.textContent        = elapsed + 's';

    $finalDwell.textContent      = features.timing.avgDwellMs + 'ms';
    $finalFlight.textContent     = features.timing.avgFlightMs + 'ms';
    $finalBurstiness.textContent = features.aggregate.burstiness;
    $finalRollovers.textContent  = features.behavior.rolloverCount;
    $finalShiftPref.textContent  = features.behavior.shiftPreference;
    $finalErrorRate.textContent  = features.aggregate.errorRate + '%';

    $typingArea.style.display    = 'none';
    $resultsScreen.style.display = 'flex';
    $restartBtn.style.display    = 'none';
    $saveStatus.textContent      = '';
    $svmLoading.style.display    = 'block';
    $svmGuess.style.display      = 'none';

    pendingSessionData = {
      name:        playerName,
      wpm,
      accuracy,
      characters:  totalTyped,
      errors:      errorCount,
      regressions: regressionCount,
      time:        elapsed,
      avgDwellMs:  features.timing.avgDwellMs,
      avgFlightMs: features.timing.avgFlightMs,
      avgDDMs:     features.timing.avgDDMs,
      avgUUMs:     features.timing.avgUUMs,
      leftShiftCount:       features.behavior.leftShiftCount,
      rightShiftCount:      features.behavior.rightShiftCount,
      shiftPreference:      features.behavior.shiftPreference,
      backspaceCount:       features.behavior.backspaceCount,
      backspaceStyle:       features.behavior.backspaceStyle,
      ctrlACount:           features.behavior.ctrlACount,
      capsLockCount:        features.behavior.capsLockCount,
      rolloverCount:        features.behavior.rolloverCount,
      avgRolloverDurationMs: features.behavior.avgRolloverDurationMs,
      errorRate:      features.aggregate.errorRate,
      burstiness:     features.aggregate.burstiness,
      wpmOverTime:    features.aggregate.wpmOverTime,
      speedDecayRate: features.aggregate.speedDecayRate,
      digraphLatencies:  _summarizeLatencyMap(features.timing.digraphLatencies),
      trigraphLatencies: _summarizeLatencyMap(features.timing.trigraphLatencies)
    };

    const predictPayload = {
      wpm, accuracy,
      characters:  totalTyped,
      errors:      errorCount,
      regressions: regressionCount,
      time:        elapsed,
      avgDwellMs:  features.timing.avgDwellMs,
      avgFlightMs: features.timing.avgFlightMs,
      avgDDMs:     features.timing.avgDDMs,
      avgUUMs:     features.timing.avgUUMs,
      leftShiftCount:       features.behavior.leftShiftCount,
      rightShiftCount:      features.behavior.rightShiftCount,
      shiftPreference:      features.behavior.shiftPreference,
      backspaceCount:       features.behavior.backspaceCount,
      backspaceStyle:       features.behavior.backspaceStyle,
      ctrlACount:           features.behavior.ctrlACount,
      capsLockCount:        features.behavior.capsLockCount,
      rolloverCount:        features.behavior.rolloverCount,
      avgRolloverDurationMs: features.behavior.avgRolloverDurationMs,
      errorRate:      features.aggregate.errorRate,
      burstiness:     features.aggregate.burstiness,
      speedDecayRate: features.aggregate.speedDecayRate,
    };

    fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(predictPayload)
    })
    .then(r => r.json())
    .then(result => {
      $svmLoading.style.display = 'none';
      if (result.error || result.prediction === null) {
        $svmSection.style.display = 'none';
        logAndSave(playerName);
        return;
      }
      svmPredictedName = result.prediction;
      $svmPrediction.textContent = svmPredictedName;
      const modelLabel = result.model_type === 'oneclass' ? 'one-class svm' : 'svc';
      $svmSubtext.textContent = result.confidence != null
        ? `confidence score: ${result.confidence}  [${modelLabel}]`
        : `[${modelLabel}]`;
      $svmGuess.style.display = 'flex';
    })
    .catch(() => {
      $svmLoading.style.display = 'none';
      $svmSection.style.display = 'none';
      logAndSave(playerName);
    });
  }

  function _summarizeLatencyMap(latencyMap) {
    const entries = Object.entries(latencyMap);
    if (!entries.length) return '';
    return entries
      .map(([ngram, values]) => {
        const avg = Math.round(values.reduce((s, v) => s + v, 0) / values.length);
        return `${ngram}:${avg}`;
      })
      .join('|');
  }

  function logAndSave(confirmedName) {
    if (!pendingSessionData) return;
    const payload = { ...pendingSessionData, name: confirmedName };

    $saveStatus.textContent = '⏳ saving...';
    $saveStatus.className   = 'save-pending';

    fetch('/api/log-session', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    })
    .then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
    .then(() => {
      $saveStatus.textContent = '✓ session saved to CSV';
      $saveStatus.className   = 'save-ok';
      fetch('/api/train', { method: 'POST' })
        .then(r => r.json())
        .then(t => console.log('Model retrained:', t))
        .catch(e => console.warn('Retrain failed:', e.message));
    })
    .catch(err => {
      $saveStatus.textContent = '✗ could not save — is the server running? (npm start)';
      $saveStatus.className   = 'save-error';
      console.warn('Session log failed:', err.message);
    })
    .finally(() => {
      $restartBtn.style.display = 'block';
      pendingSessionData = null;
    });
  }

  function handleSvmConfirm(userSaysCorrect) {
    $svmGuess.style.display = 'none';
    let confirmedName;
    if (userSaysCorrect) {
      confirmedName = svmPredictedName;
    } else {
      confirmedName = svmPredictedName === 'Joshua' ? 'Unknown' : 'Joshua';
    }
    logAndSave(confirmedName);
  }

  function logSession() {}

  // ——————————————— Prepare Test (reset state, render words) ———————————————

  function prepareTest() {
    clearInterval(timerInterval);
    clearInterval(wpmWindowTimer);
    timerStarted    = false;
    testActive      = false;
    testFinished    = false;
    timeRemaining   = TEST_TIME;
    caretIndex      = 0;
    correctCount    = 0;
    errorCount      = 0;
    regressionCount = 0;
    totalTyped      = 0;
    pendingSessionData = null;
    svmPredictedName   = null;

    collector = new KeystrokeCollector();

    $timerDisplay.textContent    = TEST_TIME;
    $wpmDisplay.textContent      = '0';
    $restartHint.classList.remove('visible');
    $restartBtn.style.display    = 'none';
    $svmLoading.style.display    = 'block';
    $svmGuess.style.display      = 'none';
    $saveStatus.textContent      = '';
    $saveStatus.className        = '';

    renderWords(pickWords(WORD_COUNT));
  }

  // ——————————————— Reset → goes back to name screen ———————————————

  function resetTest() {
    $header.style.display        = 'flex';
    $typingArea.style.display    = 'block';
    $resultsScreen.style.display = 'none';
    prepareTest();
  }

  // ——————————————— Input Handlers ———————————————

  let tabPressed = false;   // for Tab+Enter restart shortcut

  // ——— Keydown: main input handler + biometric recording ———
  document.addEventListener('keydown', (e) => {
    // If the name screen is active, let the name input handle keys natively
    if (nameScreenActive) return;

    // ——— Tab + Enter restart ———
    if (e.key === 'Tab') {
      e.preventDefault();
      tabPressed = true;
      $restartHint.classList.add('visible');
      return;
    }
    if (e.key === 'Enter' && tabPressed) {
      e.preventDefault();
      tabPressed = false;
      resetTest();
      return;
    }
    tabPressed = false;
    $restartHint.classList.remove('visible');

    // Ignore input once the test is finished (results are showing)
    if (testFinished) return;

    // ——— Record keydown in collector (ALWAYS, before any early returns) ———
    if (testActive || !timerStarted) {
      collector.recordKeyDown(e, caretIndex);
    }

    // ——— Backspace ———
    if (e.key === 'Backspace') {
      e.preventDefault();
      if (caretIndex <= 0) return;

      regressionCount++;

      // Remove caret from current position
      if (caretIndex < chars.length) {
        chars[caretIndex].classList.remove('current');
      }

      caretIndex--;

      // Remove state from the character we're stepping back to
      chars[caretIndex].classList.remove('correct', 'error');
      chars[caretIndex].classList.add('current');
      return;
    }

    // ——— Ignore non-printable / modifier keys (but collector already recorded them) ———
    if (e.key.length !== 1 || e.ctrlKey || e.altKey || e.metaKey) return;

    // ——— Start timer on first real key-press ———
    if (!timerStarted) startTimer();

    // ——— All characters typed — do nothing (timer will end test) ———
    if (caretIndex >= chars.length) return;

    e.preventDefault();

    const currentChar = chars[caretIndex];
    const expected    = currentChar.dataset.char;

    // Remove caret class
    currentChar.classList.remove('current');

    // Match check
    if (e.key === expected || (expected === ' ' && e.key === ' ')) {
      currentChar.classList.add('correct');
      correctCount++;
      collector.recordCorrectChar();
    } else {
      currentChar.classList.add('error');
      errorCount++;
    }

    totalTyped++;
    caretIndex++;

    // ——— Finished all characters → end test immediately ———
    if (caretIndex >= chars.length) {
      endTest();
      return;
    }

    // Advance caret
    chars[caretIndex].classList.add('current');
  });

  // ——— Keyup: biometric recording only ———
  document.addEventListener('keyup', (e) => {
    if (nameScreenActive) return;
    if (testFinished) return;

    collector.recordKeyUp(e);
  });

  // ——— Restart button ———
  $restartBtn.addEventListener('click', resetTest);

  // ——— SVM confirmation buttons ———
  $svmYesBtn.addEventListener('click', () => handleSvmConfirm(true));
  $svmNoBtn.addEventListener('click',  () => handleSvmConfirm(false));

  // ——————————————— Initial Load ———————————————
  // Start directly with the test
  prepareTest();

})();
