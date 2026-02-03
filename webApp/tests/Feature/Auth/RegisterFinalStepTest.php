<?php
namespace Tests\Feature\Auth;

use App\Models\Region;
use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class RegisterFinalStepTest extends TestCase
{
    use RefreshDatabase;

    private function getSessionData()
    {
        return [
            'fullname' => 'John Doe',
            'email'    => 'john@example.com',
            'dob'      => '01/01/2000', // UBAH KE FORMAT m/d/Y Sesuai Controller!
            'gender'   => '1',
            'country'  => 'ID',
        ];
    }

    public function test_registrasi_sukses_data_masuk_database()
    {
        // 1. Arrange Data
        Region::create(['id' => 'ID', 'name' => 'Indonesia', 'continent' => 'Asia']);

        $sessionData            = $this->getSessionData();
        $sessionData['country'] = 'ID';

        // 2. Act
        // Kita start session manual agar lebih persisten
        $response = $this->withSession([
            'register.step1' => $sessionData,
            'otp_passed'     => true,
        ])
            ->from(route('account')) // Simulasi user ada di halaman account
            ->post(route('auth.account'), [
                'username'              => 'johndoe123',
                'password'              => 'password123',
                'password_confirmation' => 'password123',
            ]);

        // Debugging (Opsional): Jika masih error 500, uncomment ini
        // $response->dumpSession();
        // $response->dump();

        // 3. Assert
        $response->assertRedirect(route('login'));

        $this->assertDatabaseHas('users', ['username' => 'johndoe123']);
        $this->assertDatabaseHas('user_details', ['email' => 'john@example.com']);
    }

    public function test_gagal_jika_username_sudah_terpakai()
    {
        User::factory()->create(['username' => 'taken_user']);

        $response = $this->withSession([
            'register.step1' => $this->getSessionData(),
            'otp_passed'     => true,
        ])
            ->from(route('account'))
            ->post(route('auth.account'), [
                'username'              => 'taken_user',
                'password'              => 'password123',
                'password_confirmation' => 'password123',
            ]);

        $response->assertRedirect(route('account'));
        $response->assertSessionHasErrors('username');
    }

    public function test_gagal_jika_password_tidak_cocok()
    {
        $response = $this->withSession([
            'register.step1' => $this->getSessionData(),
            'otp_passed'     => true,
        ])
            ->from(route('account'))
            ->post(route('auth.account'), [
                'username'              => 'newuser',
                'password'              => 'password123',
                'password_confirmation' => 'salah',
            ]);

        $response->assertRedirect(route('account'));
        $response->assertSessionHasErrors('password');
    }
}
