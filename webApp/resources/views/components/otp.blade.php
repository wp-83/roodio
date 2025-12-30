<div class='flex flex-col mt-4 mb-8 otp-container'>
    <div class='flex flex-row justify-center gap-4 items-center'>
        @php ($amount = 6)
        @php ($idx = 1)
        @while ($idx <= $amount)
            <input type="text" maxlength="1" inputmode="numeric" name="otp-{{ $idx }}" id="otp-{{ $idx }}" autocomplete="off" placeholder="*" class='not-placeholder-shown:bg-white bg-shadedOfGray-20 text-center text-paragraph outline-none border-2 font-bold rounded-md px-1.5 py-0.5 w-10 h-12 border-primary-30 placeholder:text-paragraph focus:border-secondary-happy-100 focus:bg-secondary-happy-20/50 ease-in-out duration-150 {{ $errors->has('otp') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}'>
            @php ($idx++)
        @endwhile
    </div>
    <div class="error-message">
        @error('otp')
            {{ $message }}
        @enderror
    </div>
</div>