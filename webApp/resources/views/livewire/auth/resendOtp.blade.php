<div class="text-micro text-center md:text-small"
     {{-- Hapus wire:init="sendOtp" di sini karena sudah ditangani mount() --}}
     x-data="{
        timer: {{ $secondsRemaining }}, {{-- Ambil nilai dinamis dari backend --}}
        canResend: {{ $secondsRemaining > 0 ? 'false' : 'true' }},
        init() {
            if (this.timer > 0) {
                this.startTimer();
            }
        },
        startTimer() {
            this.canResend = false;
            let interval = setInterval(() => {
                this.timer--;
                if (this.timer <= 0) {
                    clearInterval(interval);
                    this.timer = 0;
                    this.canResend = true;
                }
            }, 1000);
        },
        resetTimer() {
             this.timer = 60;
             this.startTimer();
        }
     }"
     @otp-resent.window="resetTimer()"
>

    <span>Don't get the code?</span>

    <button type="button"
            wire:click="sendOtp"
            x-bind:disabled="!canResend"
            wire:loading.attr="disabled"
            class="font-bold cursor-pointer transition-colors duration-200"
            :class="(!canResend) ? 'text-gray-400 cursor-not-allowed' : 'text-secondary-sad-100 hover:text-primary-50'"
    >

        <span wire:loading wire:target="sendOtp">
            <i class="fas fa-spinner fa-spin"></i> Sending...
        </span>

        <span wire:loading.remove wire:target="sendOtp">
            Resend The Code
        </span>

    </button>

    <span x-show="!canResend" class="text-gray-500 font-medium ml-1">
        in (<span x-text="timer"></span>s)
    </span>

</div>
