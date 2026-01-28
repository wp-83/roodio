<?php
namespace Tests\Feature\Auth;

use App\Models\Region;
use App\Models\User;
use App\Models\userDetails;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class ForgotPasswordTest extends TestCase
{
    use RefreshDatabase;

    /** --- STEP 1: EMAIL VERIFICATION --- */

    public function test_step1_email_valid_ditemukan()
    {
        $user = User::factory()->create(['username' => 'lupa_user']);

        $region = Region::create(['id' => 'ID', 'name' => 'Indo', 'continent' => 'Asia']);

        userDetails::create([
            'userId'      => $user->id,
            'email'       => 'lupa@gmail.com',
            'fullname'    => 'Si Lupa',
            'dateOfBirth' => '2000-01-01',
            'gender'      => 1,
            'countryId'   => $region->id,
        ]);

        $response = $this->post(route('auth.emailVerification'), [
            'email' => 'lupa@gmail.com',
        ]);

        $response->assertSessionHasNoErrors();
        $response->assertSessionHas('forgot.step1');
        $response->assertRedirect(route('user.verification'));
    }

    public function test_step1_gagal_jika_email_tidak_terdaftar()
    {
        $response = $this->post(route('auth.emailVerification'), [
            'email' => 'tidakada@gmail.com',
        ]);

        $response->assertSessionHasErrors('email');
    }

    /** --- STEP 3: RESET PASSWORD (Final) --- */

    public function test_step3_reset_password_berhasil()
    {
        // 1. Arrange Data User & Profile
        $user = User::factory()->create([
            'username' => 'reset_user',
            'password' => bcrypt('password_lama'),
        ]);

        $region = Region::create(['id' => 'ID', 'name' => 'Indo', 'continent' => 'Asia']);

        userDetails::create([
            'userId'      => $user->id,
            'email'       => 'reset@gmail.com',
            'fullname'    => 'Si Reset',
            'dateOfBirth' => '2000-01-01',
            'gender'      => 1,
            'countryId'   => $region->id,
        ]);

        // 2. Act
        // FIX: Password harus mengandung angka (password_baru123) agar lolos validasi ->numbers()
        $response = $this->withSession([
            'forgot.step1'      => ['email' => 'reset@gmail.com'],
            'otp_forgot_passed' => true,
        ])
            ->from(route('forgetPassword'))
            ->post(route('auth.forgetPassword'), [
                'password'              => 'password_baru123',
                'password_confirmation' => 'password_baru123',
            ]);

        // 3. Assert
        $response->assertRedirect(route('login'));

        // Cek login dengan password baru
        $this->assertTrue(auth()->attempt([
            'username' => 'reset_user',
            'password' => 'password_baru123',
        ]));
    }

    public function test_step3_gagal_jika_konfirmasi_password_salah()
    {
        $response = $this->withSession([
            'forgot.step1'      => ['email' => 'dummy@gmail.com'],
            'otp_forgot_passed' => true,
        ])
            ->from(route('forgetPassword'))
            ->post(route('auth.forgetPassword'), [
                'password'              => 'password_baru123',
                'password_confirmation' => 'password_beda',
            ]);

        $response->assertRedirect(route('forgetPassword'));
        $response->assertSessionHasErrors('password');
    }
}
