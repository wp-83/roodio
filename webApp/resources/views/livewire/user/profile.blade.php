<div x-data="{
    showPhotoModal: false,
    showUsernameModal: false,
    showPasswordModal: false,
    showDeleteModal: false
}"
x-on:username-updated.window="showUsernameModal = false"
x-on:photo-updated.window="showPhotoModal = false"
x-on:password-updated.window="showPasswordModal = false">

    <div class="mt-10 max-w-4xl w-[80%] mx-auto border-2 border-white bg-white rounded-3xl shadow-2xl overflow-hidden relative mb-10">

        {{-- HEADER SECTION --}}
        <div class="h-48 relative" style="background: linear-gradient(to right, 
             #FFC48D 0%, 
             #B6A5E7 33%, 
             #8EE0B1 66%, 
             #F49DA0 100%)">
            {{-- Back Button --}}
            <a href="javascript:history.back()" 
               class="absolute top-6 left-4 flex items-center gap-2 px-5 py-2 bg-primary-10 rounded-full text-white hover:bg-primary-60 transition group z-20">
                <svg xmlns="http://www.w3.org/2000/svg" 
                    class="h-5 w-5 group-hover:-translate-x-1 transition-transform stroke-primary-60 group-hover:stroke-white" 
                    fill="none" viewBox="0 0 24 24" 
                    stroke="currentColor">
                    <path stroke-linecap="round" 
                        stroke-linejoin="round" 
                        stroke-width="2" 
                        d="M10 19l-7-7m0 0l7-7m-7 7h18">
                    </path>
                </svg>
                <span class="text-smallBtn text-primary-85 font-bold group-hover:text-white">Back</span>
            </a>
            <div class='w-48 md:w-96 lg:w-md xl:w-lg absolute top-5 right-3 xl:top-1/2 xl:-translate-y-1/2'>
                <img src="{{ asset('assets/logo/logo-horizontal.png') }}" alt="logo" class='w-full'>
            </div>
        </div>

        <div class="px-8 pb-10">
            {{-- PROFILE PHOTO & SAVE BUTTON --}}
            <div class="flex flex-col md:flex-row justify-between items-end -mt-20 mb-10 gap-6 relative z-10">
                {{-- Profile Photo --}}
                <div class="relative">
                    <div class="relative">
                        @if (empty($profilePhoto))
                            <img class="w-40 h-40 rounded-full border-[6px] border-white object-cover shadow-lg bg-shadedOfGray-20"
                                 src="{{ asset('assets/defaults/user.jpg') }}"
                                 alt="Profile Photo Preview">
                        @else
                            <img class="w-40 h-40 rounded-full border-3 border-primary-50 object-cover shadow-lg bg-shadedOfGray-20"
                                 src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}"
                                 alt="Profile Photo">
                        @endif

                        {{-- Change Photo Button --}}
                        <button type="button" 
                                @click="showPhotoModal = true" 
                                class="absolute bottom-0 right-0 bg-primary-60 text-white p-2.5 rounded-full border-4 border-white hover:bg-primary-30 transition shadow-md group" 
                                title="Change Photo">
                            <svg xmlns="http://www.w3.org/2000/svg" 
                                 class="h-5 w-5 group-hover:scale-110 transition-transform" 
                                 fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                            </svg>
                        </button>
                    </div>
                </div>

                {{-- Save Changes Button --}}
                <div class="w-full md:w-auto">
                    <x-button wire:click="update"
                            disabled
                            wire:dirty.attr.remove="disabled"
                            wire:target="fullname,email,dateOfBirth,gender,countryId"
                            content='Save Changes'
                            style="zoom:0.85;">
                    </x-button>
                </div>
            </div>

            {{-- SUCCESS MESSAGE --}}
            @if (session('success'))
                <div class="flex w-full items-center p-4 mb-4 font-bold text-success-dark font-secondaryAndButton text-small md:text-body-size rounded-base bg-success-lighten/10" role="alert">
                    <svg class="w-8 h-8 me-2 shrink-0 mt-0.5 sm:mt-0" 
                         aria-hidden="true" 
                         xmlns="http://www.w3.org/2000/svg" 
                         width="24" height="24" fill="none" viewBox="0 0 24 24">
                        <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M10 11h2v5m-2 0h4m-2.592-8.5h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                    </svg>
                    <p>{{ session('success') }}</p>
                </div>
            @endif

            {{-- MAIN FORM --}}
            <form wire:submit.prevent="update">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    
                    {{-- PERSONAL INFORMATION COLUMN --}}
                    <div class="lg:col-span-2 space-y-8">
                        <div>
                            <p class="font-primary text-paragraph md:text-subtitle lg:text-title font-bold text-primary-50">Personal Information</p>
                            <p class="font-secondaryAndButton text-black text-small lg:text-body-size mb-8">Manage your personal details and public profile.</p>

                            <div class="grid grid-cols-1 md:grid-cols-2 gap-x-6">
                                {{-- Full Name --}}
                                <div class="col-span-1 md:col-span-2">
                                    <x-input id='fullname' icon='name' label='Fullname' placeholder='Ex: John Doe' wire:model="fullname" value="{{ old('fullname') }}" :isRequired='false'></x-input>
                                </div>

                                {{-- Email --}}
                                <div class="col-span-1 md:col-span-2">
                                    <x-input type='email' id='email' icon='email' label='Email Address' placeholder='Ex: john.doe321@gmail.com' value="{{ old('email') }}" wire:model="email" :isRequired='false'></x-input>
                                </div>

                                {{-- Date of Birth --}}
                                <div>
                                    <x-input type='date' id='dob' icon='date' label='Date of Birth' placeholder='mm/dd/yyyy' value="{{ old('dob') }}" wire:model="dateOfBirth" :isRequired='false'></x-input>
                                </div>

                                {{-- Gender --}}
                                <div>
                                    <x-inputSelect id='gender' icon='gender' label='Gender' class='gender-select valid:bg-accent-20/60 valid:text-shadedOfGray-100 valid:not-italic invalid:text-shadedOfGray-60 invalid:italic' defaultOption='Your gender...' wire:model="gender" required>
                                        <x-slot:options>
                                            <option value="1" {{ old('gender') === '1' ? 'selected' : '' }}>Male</option>
                                            <option value="0" {{ old('gender') === '0' ? 'selected' : '' }}>Female</option>
                                            <option value="null" {{ old('gender') === 'null' ? 'selected' : '' }}>Prefer not to say</option>
                                        </x-slot:options>
                                    </x-inputSelect>
                                </div>

                                {{-- Region --}}
                                <div class="col-span-1 md:col-span-2">
                                    <x-inputSelect id='country' icon='country' label='Country' class='country-select valid:bg-accent-20/60 valid:text-shadedOfGray-100 valid:not-italic invalid:text-shadedOfGray-60 invalid:italic' defaultOption='Your country...' wire:model="countryId" required>
                                        <x-slot:options>
                                            @forelse($regions as $region)
                                                <option value="{{ $region->id }}" {{ old('country') == $region->id ? 'selected' : '' }}>{{ $region->name }}</option>
                                            @empty
                                            @endforelse
                                        </x-slot:options>
                                    </x-inputSelect>
                                </div>

                                
                            </div>
                        </div>
                    </div>

                    {{-- ACCOUNT SECURITY COLUMN --}}
                    <div class="lg:col-span-1">
                        <div class="bg-primary-10/30 rounded-2xl p-4 h-max border-2 border-primary-50 flex flex-col justify-between">
                            
                            {{-- Security Section --}}
                            <div>
                                <div class="flex items-center gap-3 mb-8">
                                    <div class="p-2.5 bg-primary-60 border border-shadedOfGray-20 rounded-xl text-primary-100 shadow-sm">
                                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="white">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                                  d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                        </svg>
                                    </div>
                                    <h3 class="font-primary text-body-size font-bold text-primary-60">Account Security</h3>
                                </div>

                                <div class="space-y-6">
                                    {{-- Username Section --}}
                                    <div>
                                        <label class="font-secondaryAndButton block text-small lg:text-body-size font-bold text-primary-60 mb-2">Username</label>
                                        <div class="relative">
                                            <div class="flex items-center bg-white border border-shadedOfGray-20 rounded-xl px-4 py-2 pr-20">
                                                <span class="text-shadedOfGray-50 mr-2 text-small">@</span>
                                                <input type="text" 
                                                       wire:model="username" 
                                                       class="w-full outline-none text-small text-shadedOfGray-85 bg-transparent" 
                                                       readonly>
                                            </div>
                                            <button type="button" 
                                                    @click="showUsernameModal = true" 
                                                    class="absolute right-2 top-2 bottom-2 px-2 font-bold text-smallBtn text-primary-50 hover:bg-primary-10/50 rounded-md transition cursor-pointer">
                                                Edit
                                            </button>
                                        </div>
                                    </div>

                                    {{-- Password Section --}}
                                    <div>
                                        <label class="font-secondaryAndButton block text-small lg:text-body-size font-bold text-primary-60 mb-2">Password</label>
                                        <div class="relative">
                                            <input type="password" 
                                                   value="DummyPass123" 
                                                   class="w-full bg-white px-4 py-2 text-small border border-shadedOfGray-20 rounded-xl text-shadedOfGray-85 outline-none pr-20" 
                                                   readonly>
                                            <button type="button" 
                                                    @click="showPasswordModal = true" 
                                                    class="absolute right-2 top-2 bottom-2 px-2 text-smallBtn text-accent-100 hover:bg-accent-20 rounded-md font-bold transition cursor-pointer">
                                                Change
                                            </button>
                                        </div>
                                    </div>
                                    
                                    {{-- Password Info --}}
                                    <div class="">
                                    <div class="">
                                        <p class="text-micro lg:text-small text-shadedOfGray-70 italic">Last password changed: {{ $passwordLastChanged ? $passwordLastChanged->diffForHumans() : 'Never' }}</p>
                                    </div>
                                    </div>
                                </div>
                            </div>

                            {{-- DANGER ZONE - DELETE ACCOUNT --}}
                            <div class="mt-12 pt-4 border-t-3 border-primary-60">
                                <h4 class="text-error-moderate font-bold text-small md:text-body-size mb-3">Danger Zone</h4>
                                <button type="button" 
                                        @click="showDeleteModal = true" 
                                        class="w-full py-3 border border-error-moderate text-error-moderate bg-error-moderate/5 hover:bg-error-moderate/10 hover:border-error-moderate rounded-xl text-body-size transition flex items-center justify-center gap-2 font-bold cursor-pointer">
                                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                    Delete Account
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </form>

        {{-- ==================== MODALS ==================== --}}

        {{-- MODAL: PHOTO --}}
        <div x-show="showPhotoModal" 
             x-cloak 
             style="display: none;" 
             class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
            <div class="bg-white rounded-2xl w-full max-w-md mx-4 shadow-2xl overflow-hidden transform transition-all scale-100">
                <div class="p-6 bg-primary-85">
                    <h3 class="font-primary text-subtitle text-primary-10 font-bold">Update Photo</h3>
                    <p class="font-secondaryAndButton text-white text-small md:text-body-size">Select a new image for your profile.</p>
                </div>

                <div class="p-6">
                    @error('profilePhoto')
                        <div class="mb-4 p-3 rounded-lg bg-secondary-angry-10 border border-secondary-angry-20 text-secondary-angry-100 text-smallBtn flex items-start gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" 
                                      d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" 
                                      clip-rule="evenodd" />
                            </svg>
                            <span>{{ $message }}</span>
                        </div>
                    @enderror

                    <form wire:submit.prevent="uploadPhoto" enctype="multipart/form-data">
                        <div class="space-y-6">
                            {{-- Photo Preview --}}
                            <div class="flex justify-center">
                                <div class="relative w-50 h-50">
                                    @if (empty($profilePhoto))
                                        <img src="{{ asset('assets/defaults/user.jpg') }}" 
                                             class="w-full h-full rounded-full object-cover border-4 border-shadedOfGray-20 bg-shadedOfGray-10">
                                    @else
                                        <img src="{{ config('filesystems.disks.azure.url') . '/' . $profilePhoto }}" 
                                             class="w-full h-full rounded-full object-cover border-4 border-shadedOfGray-20 bg-shadedOfGray-10">
                                    @endif

                                    <label for="photoInputLivewire" 
                                           class="absolute inset-0 flex items-center justify-center bg-black/40 rounded-full opacity-0 hover:opacity-100 transition cursor-pointer">
                                        <span class="text-white text-mediumBtn">Click to Select</span>
                                    </label>
                                </div>
                            </div>

                            {{-- File Input --}}
                            <input type="file" 
                                   id="photoInputLivewire" 
                                   wire:model="photo" 
                                   class="hidden" 
                                   name="photo">

                            {{-- File Selection Info --}}
                            <div class="text-center">
                                <button type="button" 
                                        onclick="document.getElementById('photoInputLivewire').click()" 
                                        class="text-primary-50 font-secondaryAndButton text-mediumBtn hover:underline">
                                    Choose File
                                </button>
                                <p class="text-small text-shadedOfGray-60 italic">Only <b>.jpg</b> or <b>.png</b> accepted. Max 5 MB.</p>
                                <div wire:loading wire:target="photo" 
                                     class="text-primary-60 text-small italic mt-1 font-secondaryAndButton">
                                    Uploading...
                                </div>
                            </div>

                            {{-- Modal Actions --}}
                            <div class="flex gap-3 pt-2">
                                <x-button @click="showPhotoModal = false" content='Cancel' mood='grayscale' style='zoom:0.9;'></x-button>
                                <x-button actionType='submit' content='Update Photo' style='zoom:0.9;'></x-button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

        {{-- MODAL: CHANGE USERNAME --}}
        <div x-show="showUsernameModal" 
             x-cloak 
             style="display: none;" 
             class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
            <div class="bg-white rounded-2xl w-[80%] max-w-md mx-4 shadow-2xl overflow-hidden transform transition-all scale-100" 
                 @click.away="showUsernameModal = false">
                <div class="p-6 bg-primary-85">
                    <h3 class="font-primary text-paragraph lg:text-subtitle text-primary-10 font-bold">Change Username</h3>
                    <p class="font-secondaryAndButton text-white text-small md:text-body-size">Update your unique identifier.</p>
                </div>
                <div class="p-6">
                    <form wire:submit.prevent="updateUsername">
                        <div class="space-y-4">
                            {{-- New Username --}}
                            <div>
                                <label class="text-body-size font-secondaryAndButton font-bold text-primary-60">New Username</label>
                                <div class="flex items-center mt-3 bg-white border border-shadedOfGray-20 rounded-lg px-4 py-1.5 focus-within:border-primary-50 focus-within:ring-4 focus-within:ring-primary-10/40 transition">
                                    <span class="text-black mr-2 text-small">@</span>
                                    <input type="text" 
                                           wire:model.defer="newUsername" 
                                           class="w-full outline-none text-body-size text-shadedOfGray-100 placeholder-shadedOfGray-30 bg-transparent" 
                                           placeholder="new username">
                                </div>
                                @error('newUsername') 
                                    <span class="error-message">{{ $message }}</span> 
                                @enderror
                            </div>

                            {{-- Password Confirmation --}}
                            <div>
                                <label class="block text-body-size font-secondaryAndButton font-bold text-primary-60">Current Password</label>
                                <input type="password" 
                                       wire:model.defer="confirmPassword" 
                                       class="w-full px-4 py-1.5 mt-3 text-body-size border border-shadedOfGray-20 rounded-lg focus:border-primary-50 focus:ring-4 focus:ring-primary-10/40 outline-none transition text-shadedOfGray-100 bg-white" 
                                       placeholder="Confirm with password">
                                @error('confirmPassword') 
                                    <span class="error-message">{{ $message }}</span> 
                                @enderror
                            </div>

                            {{-- Modal Actions --}}
                            <div class="flex gap-3 pt-4">
                                <x-button @click="showUsernameModal = false" content='Cancel' mood='grayscale' style='zoom:0.9;'></x-button>
                                <x-button actionType='submit' content='Change' style='zoom:0.9;'></x-button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

        {{-- MODAL: CHANGE PASSWORD --}}
        <div x-show="showPasswordModal" 
             x-cloak 
             style="display: none;" 
             class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
            <div class="bg-white rounded-2xl w-full max-w-md mx-4 shadow-2xl overflow-hidden transform transition-all scale-100" 
                 @click.away="showPasswordModal = false">
                <div class="p-6 bg-primary-85">
                    <h3 class="font-primary text-paragraph lg:text-subtitle text-primary-10 font-bold">Change Password</h3>
                    <p class="font-secondaryAndButton text-white text-small md:text-body-size">Ensure your account still secure.</p>
                </div>
                
                <div class="p-6">
                    <form wire:submit.prevent="updatePassword">
                        <div class="space-y-4">
                            {{-- Current Password --}}
                            <div>
                                <label class="text-body-size font-secondaryAndButton font-bold text-primary-60">Current Password</label>
                                <div class="flex items-center mt-3 bg-white border border-shadedOfGray-20 rounded-lg px-4 py-1.5 focus-within:border-primary-50 focus-within:ring-4 focus-within:ring-primary-10/40 transition">
                                    <input type="password" 
                                           wire:model.defer="currentPassword" 
                                           class="w-full outline-none text-body-size text-shadedOfGray-100 placeholder-shadedOfGray-30 bg-transparent" 
                                           placeholder="Current Password...">
                                </div>
                                @error('currentPassword') 
                                    <span class="error-message">{{ $message }}</span> 
                                @enderror
                            </div>

                            {{-- New Password --}}
                            <div>
                                <label class="text-body-size font-secondaryAndButton font-bold text-primary-60">New Password</label>
                                <div class="flex items-center mt-3 bg-white border border-shadedOfGray-20 rounded-lg px-4 py-1.5 focus-within:border-primary-50 focus-within:ring-4 focus-within:ring-primary-10/40 transition">
                                    <input type="password" 
                                           wire:model.defer="newPassword" 
                                           class="w-full outline-none text-body-size text-shadedOfGray-100 placeholder-shadedOfGray-30 bg-transparent" 
                                           placeholder="New Password...">
                                </div>
                                @error('newPassword') 
                                    <span class="error-message">{{ $message }}</span> 
                                @enderror
                            </div>

                            {{-- Confirm New Password --}}
                            <div>
                                <label class="text-body-size font-secondaryAndButton font-bold text-primary-60">Confirm New Password</label>
                                <div class="flex items-center mt-3 bg-white border border-shadedOfGray-20 rounded-lg px-4 py-1.5 focus-within:border-primary-50 focus-within:ring-4 focus-within:ring-primary-10/40 transition">
                                    <input type="password" 
                                           wire:model.defer="newPasswordConfirmation" 
                                           class="w-full outline-none text-body-size text-shadedOfGray-100 placeholder-shadedOfGray-30 bg-transparent" 
                                           placeholder="New Password Confirmation...">
                                </div>
                                @error('newPasswordConfirmation') 
                                    <span class="error-message">{{ $message }}</span> 
                                @enderror
                            </div>

                            {{-- Modal Actions --}}
                            <div class="flex gap-3 pt-2">
                                <x-button @click="showPasswordModal = false" content='Cancel' mood='grayscale' style='zoom:0.9;'></x-button>
                                <x-button actionType='submit' content='Change' style='zoom:0.9;'></x-button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

        {{-- MODAL: DELETE ACCOUNT --}}
        <div x-show="showDeleteModal" 
             x-cloak 
             style="display: none;" 
             class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-opacity duration-300">
            <div class="bg-white rounded-2xl w-full max-w-md mx-4 shadow-2xl overflow-hidden transform transition-all scale-100 border-t-4 border-error-moderate" 
                 @click.away="showDeleteModal = false">
                <div class="bg-white p-6 pb-2">
                    {{-- Warning Icon --}}
                    <div class="w-12 h-12 rounded-full bg-error-moderate/10 flex items-center justify-center mb-4 mx-auto text-error-moderate">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                    </div>
                    
                    {{-- Warning Message --}}
                    <h3 class="font-primary text-subtitle text-shadedOfGray-100 text-center mb-2">Delete Account</h3>
                    <p class="text-shadedOfGray-70 text-small text-center font-secondaryAndButton">
                        Are you sure you want to delete your account? This action is 
                        <span class="font-bold text-error-moderate">irreversible</span>. 
                        All your data will be permanently removed.
                    </p>
                </div>

                <div class="p-6 pt-2">
                    <form wire:submit.prevent="deleteAccount">
                        <div class="space-y-4">
                            {{-- Password Confirmation --}}
                            <div>
                                <label class="block text-body-size font-secondaryAndButton text-primary-60 mb-2">
                                    Enter your password to confirm
                                </label>
                                <input type="password" 
                                       wire:model.defer="deleteConfirmationPassword" 
                                       class="w-full px-4 py-1.5 text-body-size border border-shadedOfGray-20 rounded-lg focus:border-error-moderate focus:ring-4 focus:ring-error-moderate/20 outline-none transition text-shadedOfGray-100 bg-white placeholder-shadedOfGray-30"
                                       placeholder="Your password...">
                                @error('deleteConfirmationPassword') 
                                    <span class="error-message">{{ $message }}</span> 
                                @enderror
                            </div>

                            {{-- Modal Actions --}}
                            <div class="flex gap-3 pt-4">
                                <button type="button" 
                                        @click="showDeleteModal = false" 
                                        class="flex-1 py-3 border border-shadedOfGray-20 rounded-xl text-shadedOfGray-70 font-secondaryAndButton text-body-size font-bold hover:bg-shadedOfGray-10 transition">
                                    Cancel
                                </button>
                                <button type="submit" 
                                        class="flex-1 py-3 bg-error-moderate hover:bg-red-600 text-white rounded-xl font-secondaryAndButton text-body-size font-bold shadow-md shadow-error-moderate/30 transition">
                                    Yes, Delete
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

    </div>
</div>