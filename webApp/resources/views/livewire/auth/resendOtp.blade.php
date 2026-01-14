<div class="text-micro text-center md:text-small"
     x-data="{
        timer: 60,
        canResend: false,
        init() {
            this.startTimer();
        },
        startTimer() {
            this.timer = 60;
            this.canResend = false;
            let interval = setInterval(() => {
                this.timer--;
                if (this.timer <= 0) {
                    clearInterval(interval);
                    this.canResend = true;
                }
            }, 1000);
        }
     }"
     @otp-resent.window="startTimer()"
     wire:init="sendOtp"
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
